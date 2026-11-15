"""Positional encoding (PE) ablation (paper Fig. 9).

Section III of the paper subtracts the ViT absolute positional encoding before
compression, and Fig. 9 tests that choice. `SpatioTemporalForecaster` always
subtracts PE in `forward_spatial`, so this module defines a variant that keeps
PE (models.py is left unchanged) and provides train and evaluate functions
following the same pattern as the D ablation.
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
import timm
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib.pyplot as plt

from ..config import CONFIG, device, model_save_path, run_dir, set_seed
from ..models import PatchAutoEncoder, SpatioTemporalForecaster, cleanup_gpu, normalize_image
from ..sensitivity_sweep import SequenceDataset, _load_split_data
from ..eval_setup import (
    SPLIT_IDX, TOP_K_LIST, bootstrap_ci, calculate_patch_mse_metrics, db_eval_dts,
    query_eval_dts, target_indices,
)


# =========================================================
# Keep-PE variant: subclasses models.SpatioTemporalForecaster and overrides
# only forward_spatial; everything else is identical.
# =========================================================
class SpatioTemporalForecasterKeepPE(SpatioTemporalForecaster):
    """w/ Positional Encoding arm: the PE is kept instead of subtracted."""
    def forward_spatial(self, img):
        feats = self.vit.forward_features(img)[:, 1:, :]
        # PE is not subtracted; this is the only difference from the base model.
        return self.ae.encode(feats)


def build_model_pe_variant(keep_pe, cfg=CONFIG, pretrained_vit=True, device=device):
    base_vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained_vit)
    base_vit.reset_classifier(0)
    lora_config = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                              bias="none", task_type=TaskType.FEATURE_EXTRACTION, target_modules=cfg["lora_target_modules"])
    vit = get_peft_model(base_vit, lora_config).to(device)
    ae = PatchAutoEncoder(cfg["patch_emb_size"]).to(device)
    cls = SpatioTemporalForecasterKeepPE if keep_pe else SpatioTemporalForecaster
    # temporal_dropout must come from cfg: without it the keep-PE arm trains at the
    # 0.1 default while the Proposed arm uses CONFIG (0.2), which
    # would confound the PE effect with a dropout difference.
    return cls(vit, ae, seq_len=cfg["seq_len"], embed_dim=cfg["patch_emb_size"],
               temporal_dropout=cfg.get("temporal_dropout", 0.1)).to(device)


def pe_variant_dir(keep_pe, cfg=CONFIG):
    suffix = "with_pe" if keep_pe else "without_pe"
    d = os.path.join(run_dir(cfg), f"pe_ablation_{suffix}")
    os.makedirs(d, exist_ok=True)
    return d


def pe_variant_model_path(keep_pe, cfg=CONFIG):
    return os.path.join(pe_variant_dir(keep_pe, cfg), "spatio_temporal_model.pth")


# =========================================================
# Train a PE variant (with-PE or without-PE).
# without-PE reuses the base model checkpoint (model_save_path), since the
# main pipeline already subtracts PE.
# =========================================================
def train_pe_variant(keep_pe, n_epochs=None, force_retrain=False, cfg=CONFIG):
    if not keep_pe and os.path.exists(model_save_path(cfg)):
        print("without-PE is the main pipeline (train.py) checkpoint; reusing it.")
        return model_save_path(cfg)

    ckpt_path = pe_variant_model_path(keep_pe, cfg)
    if os.path.exists(ckpt_path) and not force_retrain:
        print(f"keep_pe={keep_pe}: checkpoint already exists ({ckpt_path})")
        return ckpt_path

    n_epochs = n_epochs or cfg["n_epochs"]
    label = "with-PE" if keep_pe else "without-PE"
    print(f"\n[{label}] training ({n_epochs} epochs)...")
    set_seed(cfg["seed"])
    train_imgs_p, train_dts_p, val_imgs_p, val_dts_p = _load_split_data(cfg)
    train_ds_p = SequenceDataset(train_imgs_p, seq_len=cfg["seq_len"], pred_step=cfg["pred_step"])
    # num_workers=0: fork-worker teardown can deadlock sibling stages; data are in-RAM tensors, so workers gain nothing
    train_loader_p = DataLoader(train_ds_p, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)

    model_p = build_model_pe_variant(keep_pe, cfg, pretrained_vit=True, device=device)
    opt_p = torch.optim.AdamW([
        {'params': list(model_p.vit.parameters()) + list(model_p.ae.parameters()), 'lr': 5e-6},
        {'params': list(model_p.temporal_encoder.parameters()) + list(model_p.pred_head.parameters()) + [model_p.temporal_pos_embed], 'lr': 3e-4}
    ])
    scaler_p = GradScaler(device.type)

    for epoch in range(n_epochs):
        model_p.train()
        pbar = tqdm(train_loader_p, desc=f"[{label}] Epoch {epoch+1}/{n_epochs}")
        for seq, target, _ in pbar:
            seq, target = seq.to(device), target.to(device)
            opt_p.zero_grad()
            with autocast(device.type):
                pred_emb = model_p(seq)
                with torch.no_grad():
                    gt_emb = model_p.forward_spatial(target)
                pred_emb = F.normalize(pred_emb, p=2, dim=2, eps=1e-8)
                gt_emb = F.normalize(gt_emb, p=2, dim=2, eps=1e-8)
                B, P, Dd = pred_emb.shape
                sim = torch.matmul(pred_emb.reshape(B*P, Dd), gt_emb.reshape(B*P, Dd).T) / cfg["temperature_tau"]
                loss_nce = F.cross_entropy(sim, torch.arange(B * P).to(device))
                loss_pred = (1 - (pred_emb * gt_emb).sum(dim=2)).mean()
                loss = loss_nce + cfg["alpha"] * loss_pred
            scaler_p.scale(loss).backward()
            scaler_p.unscale_(opt_p)
            torch.nn.utils.clip_grad_norm_(model_p.parameters(), max_norm=1.0)
            scaler_p.step(opt_p); scaler_p.update()
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})

    torch.save(model_p.state_dict(), ckpt_path)
    print(f"[{label}] training done: {ckpt_path}")
    del model_p, opt_p, scaler_p, train_loader_p
    cleanup_gpu("model_p", "opt_p", "scaler_p", "train_loader_p")
    return ckpt_path


# =========================================================
# Evaluate the PE ablation and plot Fig. 9
# =========================================================
def evaluate_pe_ablation(region_key="south_korea", cfg=CONFIG):
    from ..regions import REGIONS
    curr_indices = torch.tensor(REGIONS[region_key]["indices"], dtype=torch.long, device=device)
    rows = []
    for keep_pe, label in [(False, "w/o Positional Encoding (Proposed)"), (True, "w/ Positional Encoding")]:
        ckpt_path = model_save_path(cfg) if not keep_pe else pe_variant_model_path(True, cfg)
        if not os.path.exists(ckpt_path):
            print(f"[{label}] no checkpoint ({ckpt_path}); run train_pe_variant(keep_pe={keep_pe}) first. Skipping.")
            continue

        model_p = build_model_pe_variant(keep_pe, cfg, pretrained_vit=False, device=device)
        model_p.load_state_dict(torch.load(ckpt_path, map_location=device))
        model_p.eval()

        db_patches, query_patches = [], []
        with h5py.File(cfg["paths"]["h5"], "r") as f, torch.no_grad():
            dset = f['images']
            for start_b in tqdm(range(cfg["seq_len"], SPLIT_IDX, cfg["batch_size"]), desc=f"[{label}] DB"):
                end_b = min(start_b + cfg["batch_size"], SPLIT_IDX)
                img_t = normalize_image(torch.tensor(dset[start_b:end_b], dtype=torch.float32)).to(device)
                db_patches.append(model_p.forward_spatial(img_t).index_select(1, curr_indices).cpu())
            db_patches = torch.cat(db_patches, dim=0).to(device)

            for idx in tqdm(target_indices, desc=f"[{label}] Query"):
                img_t = normalize_image(torch.tensor(dset[idx - 1], dtype=torch.float32).unsqueeze(0)).to(device)
                query_patches.append(model_p.forward_spatial(img_t).index_select(1, curr_indices).cpu())
            query_patches = torch.cat(query_patches, dim=0).to(device)

        mean_res, raw_res, _ = calculate_patch_mse_metrics(query_patches, db_patches, query_eval_dts, db_eval_dts)
        row = {"Variant": label}
        for k in TOP_K_LIST:
            row[f"Top-{k} MSE"] = mean_res[k]
            if len(raw_res[k]) > 1:
                _, lo, hi = bootstrap_ci(raw_res[k])
                row[f"Top-{k} 95% CI"] = f"[{lo:.4f}, {hi:.4f}]"
        rows.append((row, raw_res))
        del model_p, db_patches, query_patches
        cleanup_gpu("model_p", "db_patches", "query_patches")

    return rows


def pe_significance_table(raw_without, raw_with):
    """Paired Wilcoxon test, w/o-PE vs w/-PE, for every Top-k.

    Reporting all k (not only Top-1) shows whether the conclusion holds or
    flips at larger k.
    """
    rows = []
    for k in TOP_K_LIST:
        a = np.asarray(raw_without.get(k, []), dtype=np.float64)
        b = np.asarray(raw_with.get(k, []), dtype=np.float64)
        if len(a) != len(b) or len(a) < 2:
            continue
        try:
            p = stats.wilcoxon(a, b)[1]
        except ValueError:
            p = np.nan
        rows.append({
            "Top-k": k, "N": len(a),
            "w/o PE MSE": a.mean(), "w/ PE MSE": b.mean(),
            "Delta (w/o - w/)": a.mean() - b.mean(),
            "Wilcoxon p": p,
            "Significant (p<0.05)": bool(p < 0.05) if not np.isnan(p) else False,
            "Lower": "w/o PE" if a.mean() < b.mean() else "w/ PE",
        })
    return pd.DataFrame(rows)


def _plot_per_k(ax, series, title, ylabel):
    """Lines over Top-k on the x-axis with bootstrap CI bands.

    Bars are avoided (as in the unified comparison plots): the values sit in a
    narrow band near 1.0, so honest bars starting at 0 hide the differences and
    truncated bars misrepresent the axis. Lines do not mislead with a truncated
    axis.
    """
    kx = np.arange(len(TOP_K_LIST))
    for label, means, los, his, color in series:
        ax.plot(kx, means, marker="o", ms=5, lw=2, color=color, label=label)
        ax.fill_between(kx, los, his, color=color, alpha=0.15)
    ax.set_xticks(kx, [str(k) for k in TOP_K_LIST])
    ax.set_xlabel("Top-k"); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=False)


def main(cfg=CONFIG):
    # Only the with-PE arm is trained here; without-PE reuses the base model.
    train_pe_variant(keep_pe=True)

    pe_results = evaluate_pe_ablation(region_key="south_korea")
    if len(pe_results) != 2:
        print("Both variant checkpoints (w/ PE, w/o PE) are required for the comparison.")
        return

    pe_df = pd.DataFrame([r for r, _ in pe_results])
    print("\nPositional encoding ablation results:")
    print(pe_df.to_string(index=False))
    pe_df.to_csv(os.path.join(run_dir(cfg), "pe_ablation_results.csv"), index=False)

    (_, raw_without), (_, raw_with) = pe_results[0], pe_results[1]
    sig_df = pe_significance_table(raw_without, raw_with)
    print(f"\nPE ablation significance (paired Wilcoxon, Top-1 to Top-{max(TOP_K_LIST)}):")
    print(sig_df.to_string(index=False))
    sig_out = os.path.join(run_dir(cfg), "pe_ablation_significance.csv")
    sig_df.to_csv(sig_out, index=False)

    series = []
    for (label, raw), color in [(("w/o Positional Encoding (Proposed)", raw_without), "#2ca02c"),
                                 (("w/ Positional Encoding", raw_with), "#d62728")]:
        means, los, his = [], [], []
        for k in TOP_K_LIST:
            m, lo, hi = bootstrap_ci(np.asarray(raw[k], dtype=np.float64))
            means.append(m); los.append(lo); his.append(hi)
        series.append((label, means, los, his, color))

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_per_k(ax, series, "Fig. 9: Positional Encoding Ablation",
                "Weather MSE (Lower is Better)")
    for i, (_, r) in enumerate(sig_df.iterrows()):
        if r["Significant (p<0.05)"]:
            ax.annotate("*", (i, max(s[3][i] for s in series)), ha="center",
                        va="bottom", fontsize=13, color="#333")
    ax.text(0.02, 0.02, "* = paired Wilcoxon p < 0.05; bands = bootstrap 95% CI",
            transform=ax.transAxes, fontsize=8, color="#666")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir(cfg), "fig9_pe_ablation.png"), dpi=300)
    plt.close(fig)
    print("Saved: pe_ablation_results.csv, pe_ablation_significance.csv, fig9_pe_ablation.png")


if __name__ == "__main__":
    main()
