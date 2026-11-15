"""Sensitivity sweeps for alpha / tau and LoRA rank x scale (short-schedule runs).

Also provides _load_split_data and SequenceDataset, which the trained baselines reuse.
"""
import os

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import CONFIG, device, run_dir, set_seed
from .models import build_model, normalize_image


# =========================================================
# Data loading utilities
# =========================================================
def _load_split_data(cfg=CONFIG):
    with h5py.File(cfg["paths"]["h5"], "r") as f:
        imgs = f['images'][:cfg["subset_size"]]
        dts = f['datetimes'][:cfg["subset_size"]].astype(str)
    imgs_t = normalize_image(torch.tensor(imgs, dtype=torch.float32))
    split_idx = int(cfg["subset_size"] * cfg["train_val_split"])
    return imgs_t[:split_idx], dts[:split_idx], imgs_t[split_idx:], dts[split_idx:]


class SequenceDataset(Dataset):
    def __init__(self, imgs, seq_len, pred_step=1):
        self.imgs, self.seq_len, self.pred_step = imgs, seq_len, pred_step
        self.valid_indices = list(range(len(imgs) - seq_len - pred_step))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.imgs[idx:idx+self.seq_len], self.imgs[idx+self.seq_len+self.pred_step-1], idx


# =========================================================
# alpha / tau / LoRA rank sweep: one shared short train + eval routine
# =========================================================
def _short_train_eval(alpha, tau, lora_r, lora_alpha_val, train_loader, val_loader, n_epochs_sweep=3, cfg=CONFIG):
    """Short-schedule training used ONLY for relative comparison across
    hyperparameter settings, not for producing final reported numbers."""
    set_seed(cfg["seed"])
    sweep_cfg = {**cfg, "lora_r": lora_r, "lora_alpha": lora_alpha_val}
    model_s = build_model(sweep_cfg, pretrained_vit=True, device=device)
    opt_s = torch.optim.AdamW([
        {'params': list(model_s.vit.parameters()) + list(model_s.ae.parameters()), 'lr': 5e-6},
        {'params': list(model_s.temporal_encoder.parameters()) + list(model_s.pred_head.parameters()) + [model_s.temporal_pos_embed], 'lr': 3e-4}
    ])
    scaler_s = GradScaler(device.type)

    for epoch in range(n_epochs_sweep):
        model_s.train()
        for seq, target, _ in tqdm(train_loader, desc=f"[a={alpha} tau={tau} r={lora_r}] ep{epoch+1}", leave=False):
            seq, target = seq.to(device), target.to(device)
            opt_s.zero_grad()
            with autocast(device.type):
                pred_emb = model_s(seq)
                with torch.no_grad():
                    gt_emb = model_s.forward_spatial(target)
                pred_emb = F.normalize(pred_emb, p=2, dim=2, eps=1e-8)
                gt_emb = F.normalize(gt_emb, p=2, dim=2, eps=1e-8)
                B, P, D = pred_emb.shape
                sim = torch.matmul(pred_emb.reshape(B*P, D), gt_emb.reshape(B*P, D).T) / tau
                loss_nce = F.cross_entropy(sim, torch.arange(B * P).to(device))
                loss_pred = (1 - (pred_emb * gt_emb).sum(dim=2)).mean()
                loss = loss_nce + alpha * loss_pred
            scaler_s.scale(loss).backward()
            scaler_s.unscale_(opt_s)
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), max_norm=1.0)
            scaler_s.step(opt_s); scaler_s.update()

    model_s.eval()
    mses = []
    with torch.no_grad():
        for seq, target, _ in val_loader:
            seq, target = seq.to(device), target.to(device)
            pred_emb = F.normalize(model_s(seq), p=2, dim=2)
            gt_emb = F.normalize(model_s.forward_spatial(target), p=2, dim=2)
            mses.append(F.mse_loss(pred_emb, gt_emb).item())
    del model_s, opt_s, scaler_s
    torch.cuda.empty_cache()
    return float(np.mean(mses))


def plot_alpha_tau_sweep(sweep_df, save_name="alpha_tau_sweep_heatmap.png"):
    pivot = sweep_df.pivot(index="alpha", columns="tau", values="val_mse")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("tau (τ)"); ax.set_ylabel("alpha (α)"); ax.set_title("Val MSE - α × τ Sweep")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i,j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(im, ax=ax, label="Val MSE"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir(CONFIG), save_name), dpi=300); plt.show()
    best = sweep_df.iloc[0]
    print(f"Best α/τ: α={best['alpha']}, τ={best['tau']} (val_mse={best['val_mse']:.5f})")


def plot_rank_scale_grid(grid_df, save_name="rank_scale_grid_heatmap.png"):
    pivot = grid_df.pivot(index="lora_rank", columns="scale_ratio", values="val_mse")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("scale ratio (lora_alpha / r)"); ax.set_ylabel("LoRA rank (r)")
    ax.set_title("Val MSE - rank × scale Grid")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i,j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(im, ax=ax, label="Val MSE"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir(CONFIG), save_name), dpi=300); plt.show()
    best = grid_df.iloc[0]
    print(f"Best: r={best['lora_rank']}, ratio={best['scale_ratio']} "
          f"(lora_alpha={best['lora_alpha']:.0f}, val_mse={best['val_mse']:.5f})")


def main():
    train_imgs, train_dts, val_imgs, val_dts = _load_split_data()
    train_loader = DataLoader(SequenceDataset(train_imgs, CONFIG["seq_len"]), batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(SequenceDataset(val_imgs, CONFIG["seq_len"]), batch_size=CONFIG["batch_size"], shuffle=False)

    alpha_tau_rows = [{"alpha": a, "tau": t,
                        "val_mse": _short_train_eval(a, t, CONFIG["lora_r"], CONFIG["lora_alpha"], train_loader, val_loader)}
                       for a in (0.05, 0.1, 0.5, 1.0) for t in (0.05, 0.07, 0.1, 0.2)]
    sweep_df = pd.DataFrame(alpha_tau_rows).sort_values("val_mse").reset_index(drop=True)
    plot_alpha_tau_sweep(sweep_df)

    grid_rows = [{"lora_rank": r, "scale_ratio": ratio, "lora_alpha": ratio * r,
                  "val_mse": _short_train_eval(CONFIG["alpha"], CONFIG["temperature_tau"],
                                               r, ratio * r, train_loader, val_loader)}
                 for r in (4, 8, 16, 32) for ratio in (0.5, 1.0, 2.0, 4.0)]
    grid_df = pd.DataFrame(grid_rows).sort_values("val_mse").reset_index(drop=True)
    plot_rank_scale_grid(grid_df)

    sweep_df.to_csv(os.path.join(run_dir(CONFIG), "alpha_tau_sweep.csv"), index=False)
    grid_df.to_csv(os.path.join(run_dir(CONFIG), "rank_scale_grid.csv"), index=False)
    print("\nSaved: alpha_tau_sweep.csv, rank_scale_grid.csv")


if __name__ == "__main__":
    main()
