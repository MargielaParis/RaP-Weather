"""Training: InfoNCE + cosine loss training loop, search-DB construction
(t-timestamp only), and multi-region retrieval visualization."""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .config import CONFIG, device
from .models import build_model, normalize_image, get_weather_map_and_scaler
from .regions import REGIONS, region_tensor
from .reproducibility import build_repro_config
from .sensitivity_sweep import SequenceDataset, _load_split_data


def train_model(cfg=CONFIG):
    # --- Data ---
    print(f"Loading data ({cfg['subset_size']} samples)...")
    if os.path.exists(cfg["paths"]["h5"]):
        train_imgs, train_dts, val_imgs, val_dts = _load_split_data(cfg)
    else:
        print("Data not found! Using dummy data.")
        dummy = normalize_image(torch.rand(cfg["subset_size"], 3, 224, 224) * 255)
        split_idx = int(cfg["subset_size"] * cfg["train_val_split"])
        train_imgs, val_imgs = dummy[:split_idx], dummy[split_idx:]
        train_dts = np.arange(split_idx).astype(str)
        val_dts = np.arange(split_idx, cfg["subset_size"]).astype(str)

    print(f"Split ({cfg['train_val_split']:.0%} / {1-cfg['train_val_split']:.0%}):")
    print(f"   Train (DB): {len(train_imgs)} | {train_dts[0]} ~ {train_dts[-1]}")
    print(f"   Val (Query pool): {len(val_imgs)} | {val_dts[0]} ~ {val_dts[-1]}")

    train_ds = SequenceDataset(train_imgs, seq_len=cfg["seq_len"], pred_step=cfg["pred_step"])
    val_ds = SequenceDataset(val_imgs, seq_len=cfg["seq_len"], pred_step=cfg["pred_step"])
    # Shuffled batches are required: training with shuffle=False degrades retrieval weather-MSE.
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=8)

    # --- Model ---
    print("\nInitializing Model...")
    model = build_model(cfg, pretrained_vit=True, device=device)
    optimizer = torch.optim.AdamW([
        {'params': list(model.vit.parameters()) + list(model.ae.parameters()), 'lr': 5e-6},
        {'params': list(model.temporal_encoder.parameters()) + list(model.pred_head.parameters()) + [model.temporal_pos_embed], 'lr': 3e-4}
    ])
    scaler = GradScaler(device.type)

    # --- Train loop ---
    print("Training Started (InfoNCE + Cosine Loss)...")
    for epoch in range(cfg["n_epochs"]):
        model.train()
        total_loss = total_pred_loss = total_nce_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['n_epochs']}")
        for seq, target, _ in pbar:
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            with autocast(device.type):
                pred_emb = model(seq)
                with torch.no_grad():
                    gt_emb = model.forward_spatial(target)
                pred_emb = F.normalize(pred_emb, p=2, dim=2, eps=1e-8)
                gt_emb = F.normalize(gt_emb, p=2, dim=2, eps=1e-8)
                B, P, D = pred_emb.shape
                sim_matrix = torch.matmul(pred_emb.reshape(B*P, D), gt_emb.reshape(B*P, D).T) / cfg["temperature_tau"]
                labels = torch.arange(B * P).to(device)
                loss_nce = F.cross_entropy(sim_matrix, labels)
                patch_pos_sim = (pred_emb * gt_emb).sum(dim=2)
                loss_prediction = (1 - patch_pos_sim).mean()
                loss = loss_nce + (cfg["alpha"] * loss_prediction)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item(); total_pred_loss += loss_prediction.item(); total_nce_loss += loss_nce.item()
            pbar.set_postfix({'Tot': f"{loss.item():.3f}", 'InfoNCE': f"{loss_nce.item():.3f}", 'Pred(Cos)': f"{loss_prediction.item():.3f}"})
        print(f"Epoch {epoch+1} | Total: {total_loss/len(train_loader):.5f} | NCE: {total_nce_loss/len(train_loader):.5f}")

    repro_config = build_repro_config(cfg)  # refresh with actual n_epochs / final state
    return model, optimizer, scaler, train_ds, val_ds, train_dts, val_dts, train_imgs, repro_config


def build_search_db(model, train_ds, train_dts, cfg=CONFIG):
    """--- Build search DB (t-timestamp only) ---"""
    print("\nBuilding Search Database (t-timestamp Only)...")
    db_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=2)
    db_embeddings, db_datetimes = [], []
    model.eval()
    with torch.no_grad():
        for i, (seq, _, _) in enumerate(tqdm(db_loader, desc="Encoding DB (t)")):
            last_frames = seq[:, -1, :, :, :].to(device)
            embs = model.forward_spatial(last_frames)
            db_embeddings.append(embs.cpu())
            start_idx = i * 64
            for b in range(last_frames.size(0)):
                real_idx = (start_idx + b) + cfg["seq_len"] - 1
                db_datetimes.append(train_dts[real_idx])
    db_embeddings = torch.cat(db_embeddings, dim=0).to(device)
    print(f"Database Ready: {len(db_datetimes)} past scenarios")
    return db_embeddings, db_datetimes


def visualize_multi_region_retrieval(model, val_ds, val_dts, train_imgs, db_embeddings, db_datetimes,
                                      weather_map, idx_in_val, region_names=("south_korea", "low_left"),
                                      cfg=CONFIG):
    """Show top-5 retrievals per region. region_names are keys of
    regions.REGIONS, so patch indices match the canonical definitions."""
    seq, target, _ = val_ds[idx_in_val]
    last_obs_dt = val_dts[idx_in_val + cfg["seq_len"] - 1]
    target_w_vec = weather_map.get(last_obs_dt, None)
    print(f"\n{'='*60}\nQuery Input (t): {last_obs_dt}\n{'='*60}")

    seq_tensor = seq.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        last_frame = seq_tensor[:, -1, :, :, :]
        pred_emb = F.normalize(model.forward_spatial(last_frame), p=2, dim=2)
        db_embeddings_norm = F.normalize(db_embeddings, p=2, dim=2)

    for region_name in region_names:
        indices_tensor = region_tensor(region_name, device)
        bx, by, bw, bh = REGIONS[region_name]["box"]
        box_color = REGIONS[region_name]["color"]

        region_pred = pred_emb.index_select(dim=1, index=indices_tensor)
        region_db = db_embeddings_norm.index_select(dim=1, index=indices_tensor)
        sims = (region_db * region_pred).sum(dim=2).mean(dim=1)
        top_scores, top_idxs = torch.topk(sims, k=5)

        fig, axes = plt.subplots(1, 6, figsize=(20, 3.8))

        def add_box(ax):
            ax.add_patch(patches.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor=box_color, facecolor='none'))

        axes[0].imshow(seq[-1].permute(1, 2, 0).cpu().numpy())
        axes[0].set_title(f"Input Query (t)\n{last_obs_dt}", fontweight='bold', color='gray')
        add_box(axes[0]); axes[0].axis('off')

        for i, idx in enumerate(top_idxs):
            db_idx = idx.item()
            found_dt = db_datetimes[db_idx]
            found_img_idx = db_idx + cfg["seq_len"] - 1
            found_img = train_imgs[found_img_idx].permute(1, 2, 0).cpu().numpy()
            w_mse_str, title_col = "N/A", 'black'
            if target_w_vec is not None and found_dt in weather_map:
                val = mean_squared_error(target_w_vec, weather_map[found_dt])
                w_mse_str = f"{val:.4f}"
                title_col = 'blue' if val < 0.5 else ('red' if val > 2.0 else 'black')
            ax = axes[i+1]
            ax.imshow(found_img)
            ax.set_title(f"Top-{i+1}\n{found_dt}\nMSE: {w_mse_str} | Sim: {top_scores[i]:.3f}", fontsize=10, color=title_col, fontweight='bold')
            add_box(ax); ax.axis('off')
        plt.suptitle(f"Region: {region_name} | t-timestamp Matching", fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout(); plt.show()


def main():
    model, optimizer, scaler, train_ds, val_ds, train_dts, val_dts, train_imgs, repro_config = train_model()
    db_embeddings, db_datetimes = build_search_db(model, train_ds, train_dts)
    weather_map, weather_scaler = get_weather_map_and_scaler(CONFIG)

    print("\nVisualizing Results...")
    for _ in range(3):
        visualize_multi_region_retrieval(
            model, val_ds, val_dts, train_imgs, db_embeddings, db_datetimes,
            weather_map, np.random.randint(len(val_ds)),
        )
    return model, optimizer, scaler, train_ds, val_ds, train_dts, val_dts, train_imgs, db_embeddings, db_datetimes, repro_config


if __name__ == "__main__":
    main()
