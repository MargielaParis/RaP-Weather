"""Embedding diagnostics: PCA periodicity check and mean embedding value trend
(proposed vs. baselines)."""
import os
import pickle
from datetime import datetime as dt_cls

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from .config import CONFIG, device, embedding_save_path, run_dir
from .models import cleanup_gpu, load_trained_model, normalize_image
from .regions import SK_1_INDICES


def run_pca_periodicity_check(cfg=CONFIG):
    """PCA of South Korea region embeddings over time, to check for seasonal periodicity."""
    diag_model = load_trained_model(cfg=cfg, device=device)
    with h5py.File(cfg["paths"]["h5"], "r") as f:
        datetimes_full = f['datetimes'][:].astype(str)
        n_total = len(datetimes_full)

    sample_stride = 4  # sample every Nth frame to keep this diagnostic cheap
    sample_idx = np.arange(0, n_total, sample_stride)

    sampled_embs = []
    with h5py.File(cfg["paths"]["h5"], "r") as f, torch.no_grad():
        dset = f['images']
        for start in tqdm(range(0, len(sample_idx), cfg["batch_size"]), desc="Encoding for PCA"):
            batch_idx = sample_idx[start:start+cfg["batch_size"]]
            img_tensor = normalize_image(torch.tensor(dset[batch_idx.tolist()], dtype=torch.float32)).to(device)
            embs = diag_model.forward_spatial(img_tensor).cpu()
            sampled_embs.append(embs)
    sampled_embs = torch.cat(sampled_embs, dim=0)
    dates = [dt_cls.strptime(datetimes_full[i], "%Y%m%d%H%M") if len(datetimes_full[i]) >= 12 else dt_cls.now() for i in sample_idx]
    months = [d.month for d in dates]

    sk_idx = torch.tensor(SK_1_INDICES)
    vecs = sampled_embs.index_select(1, sk_idx).mean(dim=1).numpy()

    pca = PCA(n_components=2)
    vecs_pca = pca.fit_transform(vecs)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    scatter = axes[0].scatter(vecs_pca[:, 0], vecs_pca[:, 1], c=months, cmap='twilight', alpha=0.6, s=15)
    axes[0].set_title("2D PCA Projection (Colored by Month)\nChecking for Circular Seasonality", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].grid(True, linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=axes[0], ticks=range(1, 13), label='Month')

    axes[1].plot(dates, vecs_pca[:, 0], color='#40466e', linewidth=1, alpha=0.8, label='PC1 Value')
    ma = pd.Series(vecs_pca[:, 0], index=dates).rolling(window=50, center=True).mean()
    axes[1].plot(ma.index, ma.values, color='red', linewidth=2, label='Trend (Moving Avg)')
    axes[1].set_title("PC1 Value over Time\nChecking for Wave-like Periodicity", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("PC1 Value"); axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir(cfg), "periodicity_pca.png"), dpi=300)
    plt.show()
    del diag_model, sampled_embs
    cleanup_gpu("diag_model", "sampled_embs")


def run_mean_embedding_trend(cfg=CONFIG):
    """Mean South Korea embedding value over time, proposed model vs. baselines."""
    with open(embedding_save_path(cfg), "rb") as f:
        p_embs = pickle.load(f)['embeddings']
    sk_idx = torch.tensor(SK_1_INDICES)
    means_dict = {"Proposed (RaP-Weather)": p_embs.index_select(1, sk_idx).mean(dim=(1, 2)).numpy()}

    if os.path.exists(cfg["paths"]["baseline_pkl"]):
        with open(cfg["paths"]["baseline_pkl"], "rb") as f:
            baselines = pickle.load(f)
        target_group = baselines.get('south_korea', baselines)
        for model_name in ["DINOv2", "CLIP", "ConvNeXt"]:
            if model_name in target_group:
                content = target_group[model_name]
                vecs = content.get('embeddings', content.get('feats')) if isinstance(content, dict) else content
                vecs = vecs.numpy() if torch.is_tensor(vecs) else vecs
                means_dict[model_name] = vecs.mean(axis=1)

    n_models = len(means_dict)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 2.5 * n_models), sharex=True)
    axes = [axes] if n_models == 1 else axes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (model_name, mean_vals) in enumerate(means_dict.items()):
        axes[i].plot(np.arange(len(mean_vals)), mean_vals, markersize=1, linestyle='-', linewidth=0.5, alpha=0.8, color=colors[i % len(colors)])
        axes[i].set_title(f"Mean Embedding Value: {model_name}", fontsize=12, fontweight='bold')
        axes[i].set_ylabel("Mean Value"); axes[i].grid(True, linestyle='--', alpha=0.5)
    axes[-1].set_xlabel("Image Index (Chronological)")
    plt.suptitle("Embedding Mean Value Trends over Time (Region: South Korea)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir(cfg), f"mean_embeddings_comparison_D{cfg['patch_emb_size']}.png"), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    run_pca_periodicity_check()
    run_mean_embedding_trend()


if __name__ == "__main__":
    main()
