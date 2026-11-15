"""Patch embedding dimension (D) ablation (paper Fig. 8).

Sweeps D in {4, 8, 16, 32}:
1. `train_for_D(D)`: runs the full training pipeline for one D and saves the
   checkpoint and DB embeddings (same training loop as train.py).
2. `evaluate_D_ablation(D_values)`: loads each trained checkpoint and computes
   Top-k weather MSE for the Fig. 8 plot and table.

Neither function mutates CONFIG; each uses an override
`cfg_d = {**CONFIG, "patch_emb_size": D}`, so the sweep is independent of the
main pipeline.
"""
import os

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..config import CONFIG, device, embedding_save_path, model_save_path, run_dir, set_seed
from ..models import build_model, cleanup_gpu, load_trained_model, normalize_image
from ..regions import REGIONS
from ..reproducibility import build_repro_config
from ..sensitivity_sweep import SequenceDataset, _load_split_data
from ..eval_setup import (
    SPLIT_IDX, TOP_K_LIST, bootstrap_ci, calculate_patch_mse_metrics, db_eval_dts,
    query_eval_dts, target_indices,
)


# =========================================================
# Train one D (training + DB embedding save).
# Skipped if a checkpoint for that D already exists.
# =========================================================
def train_for_D(D, n_epochs=None, force_retrain=False):
    cfg_d = {**CONFIG, "patch_emb_size": D}
    ckpt_path = model_save_path(cfg_d)
    if os.path.exists(ckpt_path) and not force_retrain:
        print(f"D={D}: checkpoint already exists ({ckpt_path}); pass force_retrain=True to retrain")
        return cfg_d

    n_epochs = n_epochs or cfg_d["n_epochs"]
    print(f"\nD={D}: training ({n_epochs} epochs)...")
    set_seed(cfg_d["seed"])
    train_imgs_d, train_dts_d, val_imgs_d, val_dts_d = _load_split_data(cfg_d)
    train_ds_d = SequenceDataset(train_imgs_d, seq_len=cfg_d["seq_len"], pred_step=cfg_d["pred_step"])
    train_loader_d = DataLoader(train_ds_d, batch_size=cfg_d["batch_size"], shuffle=True, num_workers=4)

    model_d = build_model(cfg_d, pretrained_vit=True, device=device)
    opt_d = torch.optim.AdamW([
        {'params': list(model_d.vit.parameters()) + list(model_d.ae.parameters()), 'lr': 5e-6},
        {'params': list(model_d.temporal_encoder.parameters()) + list(model_d.pred_head.parameters()) + [model_d.temporal_pos_embed], 'lr': 3e-4}
    ])
    scaler_d = GradScaler(device.type)

    for epoch in range(n_epochs):
        model_d.train()
        pbar = tqdm(train_loader_d, desc=f"[D={D}] Epoch {epoch+1}/{n_epochs}")
        for seq, target, _ in pbar:
            seq, target = seq.to(device), target.to(device)
            opt_d.zero_grad()
            with autocast(device.type):
                pred_emb = model_d(seq)
                with torch.no_grad():
                    gt_emb = model_d.forward_spatial(target)
                pred_emb = F.normalize(pred_emb, p=2, dim=2, eps=1e-8)
                gt_emb = F.normalize(gt_emb, p=2, dim=2, eps=1e-8)
                B, P, Dd = pred_emb.shape
                sim = torch.matmul(pred_emb.reshape(B*P, Dd), gt_emb.reshape(B*P, Dd).T) / cfg_d["temperature_tau"]
                loss_nce = F.cross_entropy(sim, torch.arange(B * P).to(device))
                loss_pred = (1 - (pred_emb * gt_emb).sum(dim=2)).mean()
                loss = loss_nce + cfg_d["alpha"] * loss_pred
            scaler_d.scale(loss).backward()
            scaler_d.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(model_d.parameters(), max_norm=1.0)
            scaler_d.step(opt_d); scaler_d.update()
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})

    # Save DB embeddings (train split, t-timestamp only; same logic as train.py)
    os.makedirs(run_dir(cfg_d), exist_ok=True)
    torch.save(model_d.state_dict(), model_save_path(cfg_d))
    # num_workers=0: worker teardown can deadlock here; data are in-RAM tensors, so workers gain nothing
    db_loader_d = DataLoader(train_ds_d, batch_size=64, shuffle=False, num_workers=0)
    db_embs_d, db_dts_d = [], []
    model_d.eval()
    with torch.no_grad():
        for i, (seq, _, _) in enumerate(tqdm(db_loader_d, desc=f"[D={D}] Encoding DB")):
            last_frames = seq[:, -1, :, :, :].to(device)
            embs = model_d.forward_spatial(last_frames)
            db_embs_d.append(embs.cpu())
            for b in range(last_frames.size(0)):
                db_dts_d.append(train_dts_d[(i * 64 + b) + cfg_d["seq_len"] - 1])
    db_embs_d = torch.cat(db_embs_d, dim=0)
    with open(embedding_save_path(cfg_d), "wb") as f:
        import pickle
        pickle.dump({"embeddings": db_embs_d, "datetimes": db_dts_d, "model_config": build_repro_config(cfg_d)}, f)

    print(f"D={D}: training done: {model_save_path(cfg_d)}")
    del model_d, opt_d, scaler_d, db_embs_d, train_loader_d
    cleanup_gpu("model_d", "opt_d", "scaler_d", "db_embs_d", "train_loader_d")
    return cfg_d


# =========================================================
# Evaluate per-D checkpoints and plot Fig. 8
# =========================================================
def evaluate_D_ablation(D_values=(4, 8, 16, 32), region_key="south_korea", cfg=CONFIG):
    curr_indices_base = torch.tensor(REGIONS[region_key]["indices"], dtype=torch.long)
    rows, raw_by_D = [], {}
    for D in D_values:
        cfg_d = {**cfg, "patch_emb_size": D}
        ckpt_path = model_save_path(cfg_d)
        if not os.path.exists(ckpt_path):
            print(f"D={D}: no checkpoint ({ckpt_path}); run train_for_D({D}) first. Skipping.")
            continue

        model_d = load_trained_model(cfg=cfg_d, device=device)
        curr_indices = curr_indices_base.to(device)
        db_patches, query_patches = [], []
        with h5py.File(cfg["paths"]["h5"], "r") as f, torch.no_grad():
            dset = f['images']
            for start_b in tqdm(range(cfg["seq_len"], SPLIT_IDX, cfg["batch_size"]), desc=f"[D={D}] DB"):
                end_b = min(start_b + cfg["batch_size"], SPLIT_IDX)
                img_t = normalize_image(torch.tensor(dset[start_b:end_b], dtype=torch.float32)).to(device)
                db_patches.append(model_d.forward_spatial(img_t).index_select(1, curr_indices).cpu())
            db_patches = torch.cat(db_patches, dim=0).to(device)

            for idx in tqdm(target_indices, desc=f"[D={D}] Query"):
                img_t = normalize_image(torch.tensor(dset[idx - 1], dtype=torch.float32).unsqueeze(0)).to(device)
                query_patches.append(model_d.forward_spatial(img_t).index_select(1, curr_indices).cpu())
            query_patches = torch.cat(query_patches, dim=0).to(device)

        mean_res, raw_res, _ = calculate_patch_mse_metrics(query_patches, db_patches, query_eval_dts, db_eval_dts)
        row = {"D": D}
        for k in TOP_K_LIST:
            row[f"Top-{k} MSE"] = mean_res[k]
            if len(raw_res[k]) > 1:
                _, lo, hi = bootstrap_ci(raw_res[k])
                row[f"Top-{k} 95% CI"] = f"[{lo:.4f}, {hi:.4f}]"
        rows.append(row)
        raw_by_D[D] = raw_res
        del model_d, db_patches, query_patches
        cleanup_gpu("model_d", "db_patches", "query_patches")

    # Per-query raw values are returned so that differences between D values can
    # be tested for significance; means and CIs alone do not allow that.
    return pd.DataFrame(rows), raw_by_D


def d_significance_table(raw_by_D, ref_D=8):
    """Paired Wilcoxon test per D for every Top-k, against the paper's D=8.

    Pairing is valid because every D is evaluated on the same queries in the
    same order (same rationale as eval_setup.paired_significance_report).
    """
    rows = []
    base = raw_by_D.get(ref_D, {})
    for D, per_k in sorted(raw_by_D.items()):
        for k in TOP_K_LIST:
            vals = np.asarray(per_k.get(k, []), dtype=np.float64)
            if len(vals) == 0:
                continue
            mean_v, lo, hi = bootstrap_ci(vals)
            row = {"D": D, "Top-k": k, "N": len(vals), "Mean MSE": mean_v,
                   "95% CI Low": lo, "95% CI High": hi}
            base_vals = np.asarray(base.get(k, []), dtype=np.float64)
            if D != ref_D and len(base_vals) == len(vals) and len(vals) > 1:
                try:
                    p = stats.wilcoxon(base_vals, vals)[1]
                except ValueError:
                    p = np.nan
                row[f"Wilcoxon p (vs. D={ref_D})"] = p
                row["Significant (p<0.05)"] = bool(p < 0.05) if not np.isnan(p) else False
                row[f"Better than D={ref_D}"] = bool(mean_v < base_vals.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    for D in [4, 8, 16, 32]:
        train_for_D(D)

    # Only D values with an existing checkpoint are evaluated; others are skipped with a message.
    d_ablation_df, raw_by_D = evaluate_D_ablation(D_values=(4, 8, 16, 32), region_key="south_korea")
    if len(d_ablation_df) == 0:
        print("No D checkpoint available for evaluation; train each D with train_for_D(D) first.")
        return

    print("\nD ablation results:")
    print(d_ablation_df.to_string(index=False))
    d_ablation_df.to_csv(os.path.join(run_dir(CONFIG), "d_ablation_results.csv"), index=False)

    sig_df = d_significance_table(raw_by_D, ref_D=8)
    print(f"\nD ablation significance (paired Wilcoxon vs D=8, Top-1 to Top-{max(TOP_K_LIST)}):")
    print(sig_df.to_string(index=False))
    sig_df.to_csv(os.path.join(run_dir(CONFIG), "d_ablation_significance.csv"), index=False)

    # Lines with CI bands instead of bars; see pe_ablation._plot_per_k for the rationale.
    kx = np.arange(len(TOP_K_LIST))
    colors = {4: "#c7d3e8", 8: "#7f9dc9", 16: "#40466e", 32: "#1b1f33"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for D in sorted(raw_by_D):
        sub = sig_df[sig_df["D"] == D].set_index("Top-k")
        means = [sub.loc[k, "Mean MSE"] for k in TOP_K_LIST]
        los = [sub.loc[k, "95% CI Low"] for k in TOP_K_LIST]
        his = [sub.loc[k, "95% CI High"] for k in TOP_K_LIST]
        ax.plot(kx, means, marker="o", ms=5, lw=2, color=colors.get(D, "#40466e"), label=f"D={D}")
        ax.fill_between(kx, los, his, color=colors.get(D, "#40466e"), alpha=0.15)
    ax.set_xticks(kx, [str(k) for k in TOP_K_LIST])
    ax.set_xlabel("Top-k"); ax.set_ylabel("Weather MSE (Lower is Better)")
    ax.set_title("Fig. 8: Patch Embedding Dimension Ablation")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=False, title="Patch embedding dim")
    ax.text(0.02, 0.02, "bands = bootstrap 95% CI; significance vs D=8 in d_ablation_significance.csv",
            transform=ax.transAxes, fontsize=8, color="#666")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir(CONFIG), "fig8_d_ablation.png"), dpi=300)
    plt.close(fig)
    print("Saved: d_ablation_results.csv, d_ablation_significance.csv, fig8_d_ablation.png")


if __name__ == "__main__":
    main()
