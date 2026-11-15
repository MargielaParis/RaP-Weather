"""Per-region encoders at full dimension (paper Appendix B).

Code name "group2" corresponds to the paper's "per-region" encoders: one backbone
fine-tuned per region on region crops. This ablation removes the proj(->8)
compression and trains and evaluates at the backbone's native dimension, to test
whether the per-region encoders fall behind because of the 8-dimensional bottleneck.

The protocol is identical to baselines.group2_finetuned_baseline.train_group2_baseline
(only the last 20 parameter tensors are trained, InfoNCE + cosine loss, same
alpha/tau/seed/epochs/lr). The only difference is the removed proj: retrieval
vectors have feat_dim dimensions (DINOv2-S 384, CLIP-B/32 and ConvNeXtV2-T 768)
instead of 8. In the temporal module only d_model changes to feat_dim; FFN 128,
4 heads, 2 layers, dropout 0.1 and pre-norm are kept (minimal change).

Outputs are written to run_dir under new group2_fulldim_* names; existing files
are not touched. --smoke runs the full path on reduced data as a check and writes
all checkpoints and CSVs to a scratch directory.

Run from the repo root: python scripts/group2_fulldim_ablation.py
"""
import argparse
import json
import os
import tempfile
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from rap_weather.config import CONFIG, device, run_dir, set_seed
from rap_weather.models import cleanup_gpu, normalize_image
from rap_weather.regions import REGIONS
from rap_weather.sensitivity_sweep import SequenceDataset, _load_split_data
from rap_weather.eval_setup import (
    TOP_K_LIST, bootstrap_ci, calculate_mse_metrics,
    db_eval_dts, db_indices, query_eval_dts, target_indices,
)
from rap_weather.baselines.backbone_registry import (
    BACKBONE_SPECS, build_fresh_backbone, crop_and_resize,
)

REGIONS_TO_RUN = ["south_korea", "total_patches"]


class FullDimRegionPredictor(nn.Module):
    """WholeRegionPredictor with proj removed."""

    def __init__(self, backbone_name, seq_len, unfreeze_last_n_params=20):
        super().__init__()
        backbone, input_size, feat_dim = build_fresh_backbone(backbone_name)
        self.backbone = backbone
        self.input_size = input_size
        self.embed_dim = feat_dim
        for p in list(self.backbone.parameters())[:-unfreeze_last_n_params]:
            p.requires_grad = False
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, seq_len, feat_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=4, dim_feedforward=128, batch_first=True,
            dropout=0.1, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.pred_head = nn.Linear(feat_dim, feat_dim)
        nn.init.xavier_uniform_(self.pred_head.weight)
        nn.init.zeros_(self.pred_head.bias)

    def forward_spatial(self, img_batch, box):
        return self.backbone(crop_and_resize(img_batch, box, self.input_size))

    def forward(self, seq_batch, box):
        B, T, C, H, W = seq_batch.shape
        embs = self.forward_spatial(seq_batch.view(B * T, C, H, W), box).view(B, T, -1)
        embs = embs + self.temporal_pos_embed
        return self.pred_head(self.temporal_encoder(embs)[:, -1, :])


def ckpt_path(out_root, backbone_name, region_key):
    d = os.path.join(out_root, f"group2_fulldim_{backbone_name}_{region_key}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "model.pth")


def train_fulldim(backbone_name, region_key, train_loader, n_epochs, out_root, cfg=CONFIG):
    ckpt = ckpt_path(out_root, backbone_name, region_key)
    if os.path.exists(ckpt):
        print(f"[{backbone_name}/{region_key}] checkpoint exists, skipping training: {ckpt}", flush=True)
        return ckpt

    box = REGIONS[region_key]["box"]
    set_seed(cfg["seed"])
    model = FullDimRegionPredictor(backbone_name, seq_len=cfg["seq_len"]).to(device)
    opt = torch.optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': 5e-6},
        {'params': list(model.temporal_encoder.parameters())
                   + list(model.pred_head.parameters())
                   + [model.temporal_pos_embed], 'lr': 3e-4},
    ])
    scaler = GradScaler(device.type)

    print(f"[{backbone_name}/{region_key}] fulldim (D={model.embed_dim}) training "
          f"({n_epochs} epochs, {len(train_loader)} batches/epoch)", flush=True)
    for epoch in range(n_epochs):
        model.train()
        t0, tot, tot_nce, tot_pred, n = time.time(), 0.0, 0.0, 0.0, 0
        for bi, (seq, target, _) in enumerate(train_loader):
            seq, target = seq.to(device), target.to(device)
            opt.zero_grad()
            with autocast(device.type):
                pred_emb = model(seq, box)
                with torch.no_grad():
                    gt_emb = model.forward_spatial(target, box)
                pred_emb = F.normalize(pred_emb, p=2, dim=1, eps=1e-8)
                gt_emb = F.normalize(gt_emb, p=2, dim=1, eps=1e-8)
                sim = torch.matmul(pred_emb, gt_emb.T) / cfg["temperature_tau"]
                loss_nce = F.cross_entropy(sim, torch.arange(pred_emb.size(0)).to(device))
                loss_pred = (1 - (pred_emb * gt_emb).sum(dim=1)).mean()
                loss = loss_nce + cfg["alpha"] * loss_pred
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); tot_nce += loss_nce.item(); tot_pred += loss_pred.item(); n += 1
            if bi % 50 == 0:
                print(f"   [{backbone_name}/{region_key}] ep{epoch+1} b{bi}/{len(train_loader)} "
                      f"loss={loss.item():.4f} nce={loss_nce.item():.4f}", flush=True)
        print(f"[{backbone_name}/{region_key}] epoch {epoch+1}/{n_epochs} | "
              f"loss {tot/n:.5f} | nce {tot_nce/n:.5f} | pred {tot_pred/n:.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    torch.save(model.state_dict(), ckpt)
    print(f"[{backbone_name}/{region_key}] saved: {ckpt}", flush=True)
    del model, opt, scaler
    cleanup_gpu("model", "opt", "scaler")
    return ckpt


def evaluate_fulldim(backbone_name, region_key, out_root, n_db=None, n_q=None, cfg=CONFIG):
    ckpt = ckpt_path(out_root, backbone_name, region_key)
    box = REGIONS[region_key]["box"]
    model = FullDimRegionPredictor(backbone_name, seq_len=cfg["seq_len"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    def extract(indices, desc):
        indices = np.asarray(indices)
        order = np.argsort(indices)
        srt = indices[order]
        out = []
        t0 = time.time()
        with h5py.File(cfg["paths"]["h5"], "r") as f, torch.no_grad():
            dset = f["images"]
            n_batches = (len(srt) + cfg["batch_size"] - 1) // cfg["batch_size"]
            for i, st in enumerate(range(0, len(srt), cfg["batch_size"])):
                b = srt[st:st + cfg["batch_size"]]
                imgs = normalize_image(torch.tensor(dset[list(b)], dtype=torch.float32)).to(device)
                out.append(model.forward_spatial(imgs, box).cpu())
                if i % 100 == 0:
                    print(f"   [{backbone_name}/{region_key}] {desc} {i}/{n_batches} "
                          f"({time.time()-t0:.0f}s)", flush=True)
        return torch.cat(out, dim=0)[np.argsort(order)]

    db_i = db_indices[:n_db] if n_db else db_indices
    q_i = (target_indices - 1)[:n_q] if n_q else (target_indices - 1)
    d_dts = db_eval_dts[:n_db] if n_db else db_eval_dts
    q_dts = query_eval_dts[:n_q] if n_q else query_eval_dts

    db = extract(db_i, "DB")
    q = extract(q_i, "Q")
    mean_res, raw_res, _ = calculate_mse_metrics(q.to(device), db.to(device), q_dts, d_dts)
    del model, db, q
    cleanup_gpu("model", "db", "q")
    return mean_res, raw_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--regions", nargs="+", default=REGIONS_TO_RUN)
    ap.add_argument("--scratch", default=None, help="output directory for --smoke artifacts")
    args = ap.parse_args()

    t_start = time.time()
    if args.smoke:
        out_root = args.scratch or os.path.join(tempfile.gettempdir(), "g2fd_smoke")
        os.makedirs(out_root, exist_ok=True)
        n_epochs, n_db, n_q = 1, 256, 24
        cfg_data = {**CONFIG, "subset_size": 600}
    else:
        out_root = run_dir(CONFIG)
        n_epochs, n_db, n_q = CONFIG["n_epochs"], None, None
        cfg_data = CONFIG

    print(f"output root: {out_root} | epochs: {n_epochs} | regions: {args.regions}", flush=True)
    print("loading train split...", flush=True)
    train_imgs, train_dts, _, _ = _load_split_data(cfg_data)
    train_ds = SequenceDataset(train_imgs, seq_len=CONFIG["seq_len"], pred_step=CONFIG["pred_step"])
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    print(f"   train sequences: {len(train_ds)}", flush=True)

    for region in args.regions:
        rows, raws = [], {}
        for bb in BACKBONE_SPECS:
            print(f"\n{'='*55}\nfulldim per-region (code G2): {bb} / {region}\n{'='*55}", flush=True)
            train_fulldim(bb, region, train_loader, n_epochs, out_root)
            mean_res, raw_res = evaluate_fulldim(bb, region, out_root, n_db=n_db, n_q=n_q)
            raws[bb] = raw_res
            row = {"Model": bb}
            for k in TOP_K_LIST:
                v = np.asarray(raw_res[k], dtype=np.float64)
                m, lo, hi = bootstrap_ci(v)
                row[f"Top-{k}"] = m
                row[f"Top-{k} CI"] = f"[{lo:.4f}, {hi:.4f}]"
            rows.append(row)
            print(f"[{bb}/{region}] " +
                  " | ".join(f"Top-{k}: {row[f'Top-{k}']:.4f}" for k in TOP_K_LIST), flush=True)

        df = pd.DataFrame(rows).set_index("Model")
        csv_out = os.path.join(out_root, f"group2_fulldim_{region}.csv")
        df.to_csv(csv_out)
        np.savez_compressed(
            os.path.join(out_root, f"group2_fulldim_perquery_{region}.npz"),
            **{f"{bb}||Top-{k}": np.asarray(raws[bb][k], dtype=np.float64)
               for bb in raws for k in TOP_K_LIST},
        )
        print(f"\n[{region}] saved: {csv_out}\n{df.to_string()}", flush=True)

    print(f"\nall done ({(time.time()-t_start)/3600:.2f}h)", flush=True)


if __name__ == "__main__":
    main()
