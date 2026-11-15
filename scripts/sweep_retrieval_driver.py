"""Hyperparameter sweep of Appendix A, selected on the downstream task:
validation retrieval weather-MSE (south_korea region), rather than the circular
embedding-prediction MSE of the legacy sweep (rap_weather/sensitivity_sweep.py).

Protocol per config:
  1. short train (3 epochs) with the FINAL training protocol (shuffle=True)
  2. encode the full training DB (t-frames, same indices as eval_setup)
  3. encode 200 validation queries, sampled with seed 123 from the future
     pool, excluding the 500-query legacy report set
  4. score = mean weather-MSE over Top-{1,5,10} on the south_korea region

The final 5,143-query evaluation includes the entire future pool, so it includes
all 200 selection queries. The selection queries are therefore not disjoint
from the final evaluation set.

Two phases:
  phase 1: alpha x tau grid, LoRA fixed at CONFIG (r=16, lora_alpha=32)
  phase 2: rank x ratio grid at phase-1 best alpha/tau

Outputs (run dir): alpha_tau_sweep_retrieval.csv,
rank_scale_grid_retrieval.csv, heatmap PNGs, best_hparams_retrieval.json.
Outputs of the legacy embedding-MSE sweep are not overwritten.

Run from the repo root: python scripts/sweep_retrieval_driver.py
"""
import json
import os

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from rap_weather.config import CONFIG, device, run_dir, set_seed
from rap_weather.models import build_model, normalize_image
from rap_weather.regions import region_tensor
from rap_weather.sensitivity_sweep import SequenceDataset, _load_split_data
from rap_weather.eval_setup import (
    SPLIT_IDX, calculate_patch_mse_metrics, candidate_indices, db_eval_dts,
    datetimes_full, target_indices,
)

ALPHAS = (0.05, 0.1, 0.5, 1.0)
TAUS = (0.05, 0.07, 0.1, 0.2)
RANKS = (4, 8, 16, 32)
RATIOS = (0.5, 1.0, 2.0, 4.0)
N_VAL_QUERY = 200
SCORE_KS = (1, 5, 10)
N_EPOCHS_SWEEP = 3

# --- validation (model-selection) queries: disjoint from the legacy 500-query report set ---
_pool = np.setdiff1d(candidate_indices, target_indices)
VAL_TARGETS = (
    np.random.RandomState(123).choice(_pool, N_VAL_QUERY, replace=False)
    if len(_pool) >= N_VAL_QUERY
    else np.array([], dtype=np.int64)
)
VAL_QUERY_DTS = [datetimes_full[i - 1] for i in VAL_TARGETS]
SK_IDX_CPU = region_tensor("south_korea", torch.device("cpu"))


def short_train(alpha, tau, lora_r, lora_alpha_val, train_loader,
                cfg_extra=None, post_build=None, tag=""):
    """cfg_extra: extra CONFIG overrides (e.g. {"lora_dropout": 0.05}).
    post_build: callable(model) applied right after construction, for knobs that
    are baked into submodules rather than read from CONFIG (e.g. the temporal
    transformer's dropout). Both default to no-ops, so existing callers are
    unaffected and the alpha/tau and rank sweeps behave identically."""
    set_seed(CONFIG["seed"])
    cfg = {**CONFIG, "lora_r": lora_r, "lora_alpha": lora_alpha_val, **(cfg_extra or {})}
    model = build_model(cfg, pretrained_vit=True, device=device)
    if post_build is not None:
        post_build(model)
    opt = torch.optim.AdamW([
        {'params': list(model.vit.parameters()) + list(model.ae.parameters()), 'lr': 5e-6},
        {'params': list(model.temporal_encoder.parameters()) + list(model.pred_head.parameters())
                   + [model.temporal_pos_embed], 'lr': 3e-4},
    ])
    scaler = GradScaler(device.type)
    for epoch in range(N_EPOCHS_SWEEP):
        model.train()
        for seq, target, _ in tqdm(train_loader, desc=f"[a={alpha} t={tau} r={lora_r} la={lora_alpha_val}{tag}] ep{epoch+1}", leave=False):
            seq, target = seq.to(device), target.to(device)
            opt.zero_grad()
            with autocast(device.type):
                pred = model(seq)
                with torch.no_grad():
                    gt = model.forward_spatial(target)
                pred = F.normalize(pred, p=2, dim=2, eps=1e-8)
                gt = F.normalize(gt, p=2, dim=2, eps=1e-8)
                B, P, D = pred.shape
                sim = torch.matmul(pred.reshape(B*P, D), gt.reshape(B*P, D).T) / tau
                loss = F.cross_entropy(sim, torch.arange(B * P).to(device)) \
                       + alpha * (1 - (pred * gt).sum(dim=2)).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update()
    del opt, scaler
    return model


def retrieval_score(model):
    """Encode DB + validation queries, return mean SK weather-MSE over SCORE_KS."""
    model.eval()
    db_parts = []
    with h5py.File(CONFIG["paths"]["h5"], "r") as f, torch.no_grad():
        dset = f['images']
        for start in range(CONFIG["seq_len"], SPLIT_IDX, CONFIG["batch_size"]):
            end = min(start + CONFIG["batch_size"], SPLIT_IDX)
            t = normalize_image(torch.tensor(dset[start:end], dtype=torch.float32)).to(device)
            db_parts.append(model.forward_spatial(t).index_select(1, SK_IDX_CPU.to(device)).cpu())
        db = torch.cat(db_parts, dim=0)

        q_img_idx = VAL_TARGETS - 1
        order = np.argsort(q_img_idx)
        q_sorted = q_img_idx[order]
        q_parts = []
        for start in range(0, len(q_sorted), CONFIG["batch_size"]):
            batch = q_sorted[start:start + CONFIG["batch_size"]]
            t = normalize_image(torch.tensor(dset[list(batch)], dtype=torch.float32)).to(device)
            q_parts.append(model.forward_spatial(t).index_select(1, SK_IDX_CPU.to(device)).cpu())
        q = torch.cat(q_parts, dim=0)[np.argsort(order)]

    mean_res, _, _ = calculate_patch_mse_metrics(q.to(device), db.to(device),
                                                 VAL_QUERY_DTS, db_eval_dts)
    score = float(np.mean([mean_res[k] for k in SCORE_KS]))
    return score, {k: float(mean_res[k]) for k in mean_res}


def run_config(alpha, tau, lora_r, lora_alpha_val, train_loader):
    model = short_train(alpha, tau, lora_r, lora_alpha_val, train_loader)
    score, per_k = retrieval_score(model)
    del model
    torch.cuda.empty_cache()
    print(f"RSWEEP a={alpha} t={tau} r={lora_r} la={lora_alpha_val} "
          f"val_retrieval_mse={score:.4f} (Top-1 {per_k[1]:.4f})")
    return score, per_k


def heatmap(df, index, columns, title, fname):
    pivot = df.pivot(index=index, columns=columns, values="val_retrieval_mse")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel(columns); ax.set_ylabel(index); ax.set_title(title)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center",
                    color="white", fontsize=8)
    plt.colorbar(im, ax=ax, label="val retrieval weather-MSE (Top-1/5/10 mean)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir(CONFIG), fname), dpi=300)
    plt.close(fig)


def main():
    if len(VAL_TARGETS) != N_VAL_QUERY:
        raise RuntimeError(
            "Current CONFIG evaluates the full candidate pool, leaving no disjoint "
            "selection queries. Define an explicit model-selection split before rerunning."
        )
    print(f"validation queries: {N_VAL_QUERY} (seed 123, disjoint from the 500 test queries) "
          f"| score = SK weather-MSE mean over Top-{list(SCORE_KS)} | {N_EPOCHS_SWEEP}-epoch, shuffle=True")
    train_imgs, _, _, _ = _load_split_data(CONFIG)
    ds = SequenceDataset(train_imgs, seq_len=CONFIG["seq_len"], pred_step=CONFIG["pred_step"])
    train_loader = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=8)

    # ---- phase 1: alpha x tau ----
    rows = []
    for a in ALPHAS:
        for t in TAUS:
            score, per_k = run_config(a, t, CONFIG["lora_r"], CONFIG["lora_alpha"], train_loader)
            rows.append({"alpha": a, "tau": t, "val_retrieval_mse": score,
                         **{f"Top-{k}": v for k, v in per_k.items()}})
    df1 = pd.DataFrame(rows).sort_values("val_retrieval_mse").reset_index(drop=True)
    df1.to_csv(os.path.join(run_dir(CONFIG), "alpha_tau_sweep_retrieval.csv"), index=False)
    heatmap(df1, "alpha", "tau", "Val retrieval MSE: alpha x tau (task metric)",
            "alpha_tau_sweep_retrieval_heatmap.png")
    best_a, best_t = float(df1.iloc[0]["alpha"]), float(df1.iloc[0]["tau"])
    print(f"RSWEEP-BEST alpha={best_a} tau={best_t} (score={df1.iloc[0]['val_retrieval_mse']:.4f})")

    # ---- phase 2: rank x ratio at best alpha/tau ----
    rows = []
    for r in RANKS:
        for ratio in RATIOS:
            score, per_k = run_config(best_a, best_t, r, ratio * r, train_loader)
            rows.append({"lora_rank": r, "scale_ratio": ratio, "lora_alpha": ratio * r,
                         "val_retrieval_mse": score, **{f"Top-{k}": v for k, v in per_k.items()}})
    df2 = pd.DataFrame(rows).sort_values("val_retrieval_mse").reset_index(drop=True)
    df2.to_csv(os.path.join(run_dir(CONFIG), "rank_scale_grid_retrieval.csv"), index=False)
    heatmap(df2, "lora_rank", "scale_ratio", "Val retrieval MSE: rank x scale (task metric)",
            "rank_scale_grid_retrieval_heatmap.png")
    best_r = int(df2.iloc[0]["lora_rank"])
    best_la = df2.iloc[0]["lora_alpha"]
    best_la = int(best_la) if float(best_la).is_integer() else float(best_la)
    print(f"RSWEEP-BEST lora_r={best_r} lora_alpha={best_la} (score={df2.iloc[0]['val_retrieval_mse']:.4f})")

    best = {"alpha": best_a, "temperature_tau": best_t, "lora_r": best_r, "lora_alpha": best_la}
    with open(os.path.join(run_dir(CONFIG), "best_hparams_retrieval.json"), "w") as f:
        json.dump(best, f, indent=2)
    final = {"alpha": CONFIG["alpha"], "temperature_tau": CONFIG["temperature_tau"],
             "lora_r": CONFIG["lora_r"], "lora_alpha": CONFIG["lora_alpha"]}
    match = "MATCHES" if best == final else "DIFFERS FROM"
    print(f"RSWEEP-DONE winner {best} {match} final config {final}")


if __name__ == "__main__":
    main()
