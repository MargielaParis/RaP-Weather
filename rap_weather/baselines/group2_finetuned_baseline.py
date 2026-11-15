"""Paper per-region encoders (Appendix B; code name "Group 2"): predictive
fine-tuning baselines on CLIP / ConvNeXt2 / DINOv2.

Same predictive training setup as RaP-Weather ("Proposed"), but each sequence
step is a single vector from "crop the region -> encode", not a patch grid:
(B, T, 3, H, W) sequence -> crop + encode per frame -> (B, T, D) vectors ->
temporal Transformer predicts the t+n vector. The loss is the same InfoNCE +
cosine objective as RaP-Weather (CONFIG["alpha"], CONFIG["temperature_tau"],
set from the sensitivity sweep).

Differences from RaP-Weather to be aware of:
- Inputs are [0,1] tensors without per-backbone mean/std normalization.
- Temporal-encoder dropout is hardcoded to 0.1 (RaP-Weather uses 0.2).
- Retrieval uses global cosine between single vectors (calculate_mse_metrics),
  not RaP-Weather's mean per-patch cosine (calculate_patch_mse_metrics, Eq. 9).
- One model is trained per region, because the crop differs per region.
"""
import os

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import CONFIG, device, run_dir, set_seed
from ..models import cleanup_gpu, normalize_image
from ..regions import REGIONS, REGIONS_TO_EVAL
from ..sensitivity_sweep import SequenceDataset, _load_split_data
from ..eval_setup import TOP_K_LIST, calculate_mse_metrics, db_eval_dts, db_indices, query_eval_dts, target_indices
from ..eval_mse_retrieval import FINAL_REPORT, FINAL_REPORT_RAW, FINAL_REPORT_TOPIDX
from .backbone_registry import BACKBONE_SPECS, build_fresh_backbone, crop_and_resize


# =========================================================
# Per-region model: crop-encode backbone + predictive temporal Transformer.
# Same temporal modeling as models.SpatioTemporalForecaster; the difference is
# that each step is one region vector instead of 196 patch embeddings.
# =========================================================
class WholeRegionPredictor(nn.Module):
    def __init__(self, backbone_name, embed_dim, seq_len, unfreeze_last_n_params=20):
        super().__init__()
        backbone, input_size, feat_dim = build_fresh_backbone(backbone_name)  # fresh weights, independent of the frozen
        self.backbone = backbone                                              # get_frozen_backbone cache (no sharing with Group 1)
        self.input_size = input_size
        for p in list(self.backbone.parameters())[:-unfreeze_last_n_params]:
            p.requires_grad = False  # fine-tune only the last layers (similar intent to RaP-Weather's LoRA)

        self.proj = nn.Sequential(nn.Linear(feat_dim, 256), nn.GELU(), nn.Linear(256, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        # dropout=0.1 is hardcoded here; RaP-Weather's temporal encoder uses 0.2.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.1, norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.pred_head = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.pred_head.weight); nn.init.zeros_(self.pred_head.bias)

    def forward_spatial(self, img_batch, box):
        """(B,3,224,224) -> region crop -> (B, embed_dim). Same name and role as
        models.SpatioTemporalForecaster.forward_spatial, but returns one vector
        instead of 196 patch embeddings."""
        crop = crop_and_resize(img_batch, box, self.input_size)
        feat = self.backbone(crop)
        return self.proj(feat)

    def forward(self, seq_batch, box):
        """(B, T, 3, H, W) -> (B, embed_dim) predicted t+1 vector."""
        B, T, C, H, W = seq_batch.shape
        flat = seq_batch.view(B * T, C, H, W)
        embs = self.forward_spatial(flat, box).view(B, T, -1)
        embs = embs + self.temporal_pos_embed
        t_out = self.temporal_encoder(embs)
        return self.pred_head(t_out[:, -1, :])


def group2_checkpoint_path(backbone_name, region_key="south_korea", cfg=CONFIG):
    d = os.path.join(run_dir(cfg), f"group2_{backbone_name}_{region_key}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "model.pth")


def train_group2_baseline(backbone_name, region_key="south_korea", n_epochs=None,
                            force_retrain=False, cfg=CONFIG):
    ckpt_path = group2_checkpoint_path(backbone_name, region_key, cfg)
    if os.path.exists(ckpt_path) and not force_retrain:
        print(f"[{backbone_name}] already trained: {ckpt_path}")
        return ckpt_path

    n_epochs = n_epochs or cfg["n_epochs"]
    box = REGIONS[region_key]["box"]
    print(f"\n[Group2-{backbone_name}] training ({n_epochs} epochs, region={region_key})...")

    set_seed(cfg["seed"])
    train_imgs_g, train_dts_g, _, _ = _load_split_data(cfg)
    train_ds_g = SequenceDataset(train_imgs_g, seq_len=cfg["seq_len"], pred_step=cfg["pred_step"])
    train_loader_g = DataLoader(train_ds_g, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)

    model_g = WholeRegionPredictor(backbone_name, embed_dim=cfg["patch_emb_size"], seq_len=cfg["seq_len"]).to(device)
    # Separate learning rate for the temporal part, as in RaP-Weather.
    opt_g = torch.optim.AdamW([
        {'params': [p for p in model_g.backbone.parameters() if p.requires_grad] + list(model_g.proj.parameters()), 'lr': 5e-6},
        {'params': list(model_g.temporal_encoder.parameters()) + list(model_g.pred_head.parameters()) + [model_g.temporal_pos_embed], 'lr': 3e-4}
    ])
    scaler_g = GradScaler(device.type)

    for epoch in range(n_epochs):
        model_g.train()
        pbar = tqdm(train_loader_g, desc=f"[Group2-{backbone_name}] Epoch {epoch+1}/{n_epochs}")
        for seq, target, _ in pbar:
            seq, target = seq.to(device), target.to(device)
            opt_g.zero_grad()
            with autocast(device.type):
                pred_emb = model_g(seq, box)                       # (B, D)
                with torch.no_grad():
                    gt_emb = model_g.forward_spatial(target, box)  # (B, D)
                pred_emb = F.normalize(pred_emb, p=2, dim=1, eps=1e-8)
                gt_emb = F.normalize(gt_emb, p=2, dim=1, eps=1e-8)
                sim = torch.matmul(pred_emb, gt_emb.T) / cfg["temperature_tau"]
                loss_nce = F.cross_entropy(sim, torch.arange(pred_emb.size(0)).to(device))
                loss_pred = (1 - (pred_emb * gt_emb).sum(dim=1)).mean()
                loss = loss_nce + cfg["alpha"] * loss_pred
            scaler_g.scale(loss).backward()
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=1.0)
            scaler_g.step(opt_g); scaler_g.update()
            pbar.set_postfix({'loss': f"{loss.item():.3f}", 'nce': f"{loss_nce.item():.3f}", 'pred': f"{loss_pred.item():.3f}"})

    torch.save(model_g.state_dict(), ckpt_path)
    print(f"[Group2-{backbone_name}] training done: {ckpt_path}")
    del model_g, opt_g, scaler_g, train_loader_g
    cleanup_gpu("model_g", "opt_g", "scaler_g", "train_loader_g")
    return ckpt_path


def evaluate_group2(backbone_name, region_key="south_korea", cfg=CONFIG):
    ckpt_path = group2_checkpoint_path(backbone_name, region_key, cfg)
    if not os.path.exists(ckpt_path):
        print(f"[Group2-{backbone_name}/{region_key}] checkpoint not found ({ckpt_path}); run train_group2_baseline('{backbone_name}', region_key='{region_key}') first. Skipping.")
        return None, None, None

    box = REGIONS[region_key]["box"]
    model_g = WholeRegionPredictor(backbone_name, embed_dim=cfg["patch_emb_size"], seq_len=cfg["seq_len"]).to(device)
    model_g.load_state_dict(torch.load(ckpt_path, map_location=device))
    model_g.eval()

    def extract(indices, desc):
        # h5py fancy indexing requires increasing indices: sort, read, then restore order.
        indices = np.asarray(indices)
        order = np.argsort(indices)
        sorted_indices = indices[order]

        embs_sorted = []
        with h5py.File(cfg["paths"]["h5"], "r") as f, torch.no_grad():
            dset = f['images']
            for start in tqdm(range(0, len(sorted_indices), cfg["batch_size"]), desc=f"[Group2-{backbone_name}/{region_key}] {desc}"):
                batch_idx = sorted_indices[start:start + cfg["batch_size"]]
                imgs = normalize_image(torch.tensor(dset[list(batch_idx)], dtype=torch.float32)).to(device)
                embs_sorted.append(model_g.forward_spatial(imgs, box).cpu())
        embs_sorted = torch.cat(embs_sorted, dim=0)

        inverse_order = np.argsort(order)
        return embs_sorted[inverse_order]

    db_embs = extract(db_indices, "DB")
    query_embs = extract((target_indices - 1), "Query")
    mean_res, raw_res, top_idx = calculate_mse_metrics(query_embs.to(device), db_embs.to(device), query_eval_dts, db_eval_dts)

    model_key = f"{backbone_name} (Group2, predictive fine-tuned)"
    if region_key in FINAL_REPORT_RAW:
        FINAL_REPORT[region_key][model_key] = mean_res
        FINAL_REPORT_RAW[region_key][model_key] = raw_res
        FINAL_REPORT_TOPIDX[region_key][model_key] = top_idx

    del model_g, db_embs, query_embs
    cleanup_gpu("model_g", "db_embs", "query_embs")
    return mean_res, raw_res, top_idx


def main(cfg=CONFIG):
    # Train one model per backbone x region.
    # This baseline is crop-and-recompute, so each region needs its own model trained
    # on that crop, unlike RaP-Weather (trained once, then indexed per region).
    for backbone_name in BACKBONE_SPECS.keys():
        for region_key in REGIONS_TO_EVAL:
            train_group2_baseline(backbone_name, region_key=region_key, cfg=cfg)

    # Evaluate every region x backbone pair that has a trained checkpoint.
    group2_results = {}
    for region_key in REGIONS_TO_EVAL:
        group2_results[region_key] = {}
        for backbone_name in BACKBONE_SPECS.keys():
            print(f"\n{'='*50}\nGroup 2 (per-region) - {backbone_name} / {region_key}\n{'='*50}")
            mean_res, raw_res, top_idx = evaluate_group2(backbone_name, region_key=region_key, cfg=cfg)
            if mean_res is not None:
                group2_results[region_key][backbone_name] = mean_res
                for k in TOP_K_LIST:
                    print(f"   Top-{k}: MSE = {mean_res[k]:.4f}")

    for region_key, res in group2_results.items():
        if res:
            group2_df = pd.DataFrame(res).T
            group2_df.columns = [f"Top-{k}" for k in TOP_K_LIST]
            print(f"\n[{region_key}] Group 2 (per-region) summary:")
            print(group2_df)
            group2_df.to_csv(os.path.join(run_dir(cfg), f"group2_finetuned_{region_key}.csv"))
            print(f"Saved: group2_finetuned_{region_key}.csv")
        else:
            print(f"[{region_key}] no trained Group 2 checkpoints found.")

    return group2_results


if __name__ == "__main__":
    main()
