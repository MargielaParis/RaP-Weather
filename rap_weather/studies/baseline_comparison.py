"""Additional baselines beyond the off-the-shelf pretrained features.

The off-the-shelf baselines (ConvNeXt/CLIP/DINOv2, see eval_mse_retrieval.py)
are not fine-tuned, so a trained model is expected to beat them. This module
adds:

1. Pixel-NN: a training-free nearest-neighbor search in raw pixel space, the
   non-learned comparison closest to classical weather-analog retrieval.
2. ConvNeXt-FT: ConvNeXt fine-tuned with the same patch-level InfoNCE as the
   proposed model, to separate the effect of the architecture from the effect
   of training.

   ConvNeXt-FT is a backbone-swap ablation, not a baseline. The baselines in
   baselines/ (group1/2/3) cover the "crop the region into one vector" side.
   ConvNeXt-FT sits on the same side as Proposed (patch query, trained on the
   whole image) and only swaps the backbone ViT -> ConvNeXt. A fine-tuned
   comparison with the same objective is already provided by the per-region
   encoders (code group2, paper Appendix B: same InfoNCE + cosine, trained per
   region).

   ConvNeXt-FT is therefore not merged into FINAL_REPORT*. Merging would place
   it next to the per-region encoders as a baseline in the significance,
   unified, seasonal, and fig_unified_g123 outputs, which is wrong by design.
   evaluate_convnext_finetuned() writes dedicated tables instead:
     convnext_ft_ablation_<region>.csv   per-k MSE + CI, comparison with
                                         Proposed, and Wilcoxon vs. the
                                         same-backbone ConvNeXt2 (group2)
     convnext_ft_seasonal_<region>.csv   per-k seasonal MSE

   Caveat: ConvNeXt-FT differs from Proposed not only in the backbone but also
   in (a) no temporal Transformer, (b) no alpha * cosine term, (c) patch grid
   196 -> 49, and (d) last-stage unfreezing instead of LoRA. The difference
   cannot be attributed to the backbone alone. The clean comparison is
   ConvNeXt-FT vs. ConvNeXt2 (group2), which share the backbone and isolate the
   gain of patch-level representations.
3. Retrieval comparison for an arbitrary raw index, independent of the
   500-query sample.

Methods from related work are reproduced only when public code exists;
otherwise their reported numbers can be cited only if the dataset matches.
"""
import os

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from scipy import stats
from sklearn.metrics import mean_squared_error
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..config import CONFIG, device, run_dir
from ..models import PatchAutoEncoder, cleanup_gpu, load_trained_model, normalize_image
from ..regions import REGIONS, get_bbox_from_indices
from ..sensitivity_sweep import SequenceDataset, _load_split_data
from ..eval_setup import (
    TOP_K_LIST, bootstrap_ci, db_eval_dts, db_indices, query_eval_dts, target_indices, weather_map,
)
from ..eval_mse_retrieval import FINAL_REPORT_RAW


# =========================================================
# Classical baseline: nearest-neighbor search in raw pixel space
# (no fine-tuning; closest to classical weather-analog retrieval)
# =========================================================
def calculate_pixel_space_mse_metrics(query_indices, db_index_range, target_dts, db_dts,
                                        region_key="south_korea", downsample_to=32, cfg=CONFIG):
    """Flatten the (downsampled) image crop and retrieve by cosine similarity.
    Pure pixel baseline with no learned representation."""
    box = REGIONS[region_key]["box"]
    cx, cy, cw, ch = box

    def load_and_flatten(idx):
        with h5py.File(cfg["paths"]["h5"], "r") as f:
            img = f['images'][idx]
        if img.shape[0] == 3: img = img.transpose(1, 2, 0)
        roi = img[cy:cy+ch, cx:cx+cw, :]
        roi_small = np.array(F.interpolate(
            torch.tensor(roi).permute(2, 0, 1).unsqueeze(0).float(),
            size=(downsample_to, downsample_to), mode='bilinear', align_corners=False
        )).flatten()
        return roi_small

    db_vecs = np.stack([load_and_flatten(i) for i in tqdm(db_index_range, desc="[Pixel-NN] DB")])
    db_vecs = db_vecs / (np.linalg.norm(db_vecs, axis=1, keepdims=True) + 1e-8)

    results = {k: [] for k in TOP_K_LIST}
    for q_idx, t_dt in zip(query_indices, target_dts):
        if t_dt not in weather_map: continue
        q_vec = load_and_flatten(q_idx - 1)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        sims = db_vecs @ q_vec
        top_idxs = np.argsort(-sims)[:max(TOP_K_LIST)]
        target_w = weather_map[t_dt]
        for k in TOP_K_LIST:
            mses = [mean_squared_error(target_w, weather_map[db_dts[di]]) for di in top_idxs[:k] if db_dts[di] in weather_map]
            if mses: results[k].append(np.mean(mses))
    mean_results = {k: (np.mean(v) if v else np.nan) for k, v in results.items()}
    return mean_results, results


def run_pixel_nn_baseline(cfg=CONFIG):
    # Pixel similarity over the full DB; can be slower than the learned methods
    pixel_mean, pixel_raw = calculate_pixel_space_mse_metrics(
        target_indices, db_indices, query_eval_dts, db_eval_dts, region_key="south_korea", cfg=cfg
    )
    print("Pixel-space Nearest Neighbor baseline (no learning):")
    for k in TOP_K_LIST:
        print(f"   Top-{k}: MSE = {pixel_mean[k]:.4f}")

    # Significance test vs. Proposed (reuses eval_mse_retrieval results)
    if "south_korea" in FINAL_REPORT_RAW:
        proposed_raw = FINAL_REPORT_RAW["south_korea"].get("Proposed", {})
        for k in TOP_K_LIST:
            if k in proposed_raw and len(proposed_raw[k]) == len(pixel_raw[k]) and len(pixel_raw[k]) > 1:
                _, p = stats.wilcoxon(proposed_raw[k], pixel_raw[k])
                print(f"   Top-{k} vs Proposed: Wilcoxon p={p:.5f} ({'significant' if p < 0.05 else 'not significant'})")

    pd.DataFrame([{"Model": "Pixel-NN (no learning)", **{f"Top-{k}": pixel_mean[k] for k in TOP_K_LIST}}]) \
      .to_csv(os.path.join(run_dir(cfg), "pixel_nn_baseline.csv"), index=False)
    print(f"\nSaved: pixel_nn_baseline.csv")
    return pixel_mean, pixel_raw


# =========================================================
# Fine-tuned ConvNeXt trained with the same InfoNCE as the proposed model
# (separates the architecture effect from the fine-tuning effect)
# =========================================================
class ConvNeXtSpatialEncoder(nn.Module):
    """Compress ConvNeXt-Tiny features with the same PatchAutoEncoder as the
    proposed model. Fine-tunes only the last stage instead of LoRA, since
    ConvNeXt has different LoRA target modules."""
    def __init__(self, patch_emb_size, freeze_backbone_until_last_stage=True):
        super().__init__()
        backbone = tv_models.convnext_tiny(weights=tv_models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.features = backbone.features
        if freeze_backbone_until_last_stage:
            for p in list(self.features.parameters())[:-20]:
                p.requires_grad = False
        self.ae = PatchAutoEncoder(patch_emb_size)  # reused from models.py
        self.out_channels = 768  # channels of the last convnext_tiny stage

    def forward_spatial(self, img):
        feat_map = self.features(img)               # (B, 768, 7, 7)
        B, C, H, W = feat_map.shape
        patches_ = feat_map.flatten(2).transpose(1, 2)  # (B, 49, 768); 7x7=49 grid, unlike the 196-patch ViT grid
        return self.ae.encode(patches_)               # (B, 49, D)


def train_finetuned_convnext_baseline(n_epochs=None, force_retrain=False, cfg=CONFIG):
    """Fine-tune ConvNeXt with the same InfoNCE + cosine loss as the proposed model.
    Compared with the off-the-shelf ConvNeXt baseline, this separates gains
    due to the architecture from gains due to training."""
    ckpt_path = os.path.join(run_dir(cfg), "convnext_finetuned_baseline.pth")
    if os.path.exists(ckpt_path) and not force_retrain:
        print(f"Already trained: {ckpt_path}")
        return ckpt_path

    n_epochs = n_epochs or cfg["n_epochs"]
    print(f"\nTraining ConvNeXt fine-tuned baseline ({n_epochs} epochs)...")
    print("   NOTE: this baseline uses the ConvNeXt 7x7=49 grid instead of the 196 ViT patches,")
    print("   so REGIONS indices must be remapped to the ConvNeXt grid for comparison (see grid_index_remap).")

    train_imgs_c, train_dts_c, val_imgs_c, val_dts_c = _load_split_data(cfg)
    train_ds_c = SequenceDataset(train_imgs_c, seq_len=cfg["seq_len"], pred_step=cfg["pred_step"])
    train_loader_c = DataLoader(train_ds_c, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)

    model_c = ConvNeXtSpatialEncoder(cfg["patch_emb_size"]).to(device)
    opt_c = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_c.parameters()), lr=5e-6)
    scaler_c = GradScaler(device.type)

    for epoch in range(n_epochs):
        model_c.train()
        pbar = tqdm(train_loader_c, desc=f"[ConvNeXt-FT] Epoch {epoch+1}/{n_epochs}")
        for seq, target, _ in pbar:
            anchor, target = seq[:, -1, :, :, :].to(device), target.to(device)
            opt_c.zero_grad()
            with autocast(device.type):
                a_emb = F.normalize(model_c.forward_spatial(anchor), p=2, dim=2, eps=1e-8)
                t_emb = F.normalize(model_c.forward_spatial(target), p=2, dim=2, eps=1e-8)
                B, P, Dd = a_emb.shape
                sim = torch.matmul(a_emb.reshape(B*P, Dd), t_emb.reshape(B*P, Dd).T) / cfg["temperature_tau"]
                loss = F.cross_entropy(sim, torch.arange(B * P).to(device))
            scaler_c.scale(loss).backward()
            scaler_c.step(opt_c); scaler_c.update()
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})

    torch.save(model_c.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")
    del model_c, opt_c, scaler_c, train_loader_c
    cleanup_gpu("model_c", "opt_c", "scaler_c", "train_loader_c")
    return ckpt_path


# =========================================================
# Patch index remapping between the ViT 14x14 grid and the ConvNeXt 7x7 grid
# ViT patch = 16 px, convnext_tiny final feature-map cell = 32 px
# -> each 7x7 cell covers exactly one 2x2 block of the 14x14 grid (32/16 = 2)
# =========================================================
def grid_index_remap(vit_indices, vit_grid=14, target_grid=7):
    """Map ViT grid indices to the overlapping ConvNeXt (target) grid indices
    (deduplicated, sorted). Fails if vit_grid is not divisible by target_grid,
    since the two grids are then incompatible."""
    assert vit_grid % target_grid == 0, (
        f"vit_grid ({vit_grid}) is not divisible by target_grid ({target_grid}); "
        f"patch boundaries do not align, so this remapping cannot be used."
    )
    factor = vit_grid // target_grid
    mapped = set()
    for idx in vit_indices:
        row, col = idx // vit_grid, idx % vit_grid
        t_row, t_col = row // factor, col // factor
        mapped.add(t_row * target_grid + t_col)
    return sorted(mapped)


def build_convnext_regions():
    """Rebuild REGIONS (regions.py) on the ConvNeXt grid."""
    regions_convnext = {
        name: {
            "indices": grid_index_remap(r["indices"], vit_grid=14, target_grid=7),
            "box": get_bbox_from_indices(
                grid_index_remap(r["indices"], vit_grid=14, target_grid=7),
                grid_size=7, patch_size=32,
            ),
        }
        for name, r in REGIONS.items()
    }
    for name, r in regions_convnext.items():
        print(f"{name:18s} | ViT {len(REGIONS[name]['indices'])} patches -> ConvNeXt {len(r['indices'])} patches | box={r['box']}")
    return regions_convnext


REGIONS_CONVNEXT = build_convnext_regions()


# =========================================================
# ConvNeXt-FT quantitative evaluation, written to dedicated tables only.
# Not merged into the shared FINAL_REPORT* (see the module docstring).
# =========================================================
CONVNEXT_FT_LABEL = "ConvNeXt-FT (patch InfoNCE, no temporal module)"
# Same backbone (ConvNeXt) trained as "region crop -> one vector" (code group2,
# per-region encoder). Compare against this, not Proposed, to read the gain of
# patch-level representations with the backbone held fixed.
SAME_BACKBONE_REF = "ConvNeXt2 (Group2, predictive fine-tuned)"


def evaluate_convnext_finetuned(region_key="south_korea", cfg=CONFIG):
    from ..eval_setup import calculate_patch_mse_metrics, SPLIT_IDX

    ckpt_path = os.path.join(run_dir(cfg), "convnext_finetuned_baseline.pth")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found ({ckpt_path}). Run train_finetuned_convnext_baseline() first.")
        return None, None, None

    model_c = ConvNeXtSpatialEncoder(cfg["patch_emb_size"]).to(device)
    model_c.load_state_dict(torch.load(ckpt_path, map_location=device))
    model_c.eval()
    curr_indices = torch.tensor(REGIONS_CONVNEXT[region_key]["indices"], dtype=torch.long, device=device)

    db_patches, query_patches = [], []
    with h5py.File(cfg["paths"]["h5"], "r") as f, torch.no_grad():
        dset = f['images']
        for start_b in tqdm(range(cfg["seq_len"], SPLIT_IDX, cfg["batch_size"]), desc="[ConvNeXt-FT] DB"):
            end_b = min(start_b + cfg["batch_size"], SPLIT_IDX)
            img_t = normalize_image(torch.tensor(dset[start_b:end_b], dtype=torch.float32)).to(device)
            db_patches.append(model_c.forward_spatial(img_t).index_select(1, curr_indices).cpu())
        db_patches = torch.cat(db_patches, dim=0).to(device)

        for i in tqdm(target_indices, desc="[ConvNeXt-FT] Query"):
            img_t = normalize_image(torch.tensor(dset[i - 1], dtype=torch.float32).unsqueeze(0)).to(device)
            query_patches.append(model_c.forward_spatial(img_t).index_select(1, curr_indices).cpu())
        query_patches = torch.cat(query_patches, dim=0).to(device)

    mean_res, raw_res, top_idx = calculate_patch_mse_metrics(query_patches, db_patches, query_eval_dts, db_eval_dts)
    del model_c, db_patches, query_patches
    cleanup_gpu("model_c", "db_patches", "query_patches")

    write_convnext_ft_tables(mean_res, raw_res, top_idx, region_key=region_key, cfg=cfg)
    return mean_res, raw_res, top_idx


def _paired_p(a, b):
    """Paired Wilcoxon test. NaN if lengths differ or there are too few samples."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        return np.nan
    try:
        return stats.wilcoxon(a, b)[1]
    except ValueError:      # all differences zero: test undefined
        return np.nan


def write_convnext_ft_tables(mean_res, raw_res, top_idx, region_key="south_korea", cfg=CONFIG):
    """Write the two ConvNeXt-FT tables. FINAL_REPORT* is not modified.

    Two reference points are reported:
      vs Proposed      - backbone, temporal module, loss, and grid all differ,
                         so the cause cannot be isolated
      vs ConvNeXt2 G2  - same backbone, so the difference reflects patch-level
                         representation + temporal alignment
    """
    ref_raw = FINAL_REPORT_RAW.get(region_key, {})
    prop_raw = ref_raw.get("Proposed", {})
    same_raw = ref_raw.get(SAME_BACKBONE_REF, {})

    rows = []
    for k in TOP_K_LIST:
        vals = np.asarray(raw_res.get(k, []), dtype=np.float64)
        if len(vals) == 0:
            continue
        mean_v, lo, hi = bootstrap_ci(vals)
        p_prop = _paired_p(prop_raw.get(k, []), vals)
        p_same = _paired_p(same_raw.get(k, []), vals)
        prop_mean = float(np.mean(prop_raw[k])) if k in prop_raw and len(prop_raw[k]) else np.nan
        same_mean = float(np.mean(same_raw[k])) if k in same_raw and len(same_raw[k]) else np.nan
        rows.append({
            "Top-k": k, "N": len(vals),
            "ConvNeXt-FT MSE": mean_v, "95% CI Low": lo, "95% CI High": hi,
            "Proposed MSE": prop_mean,
            "Delta vs Proposed": mean_v - prop_mean,
            "Wilcoxon p (vs. Proposed)": p_prop,
            "Significant vs Proposed": bool(p_prop < 0.05) if not np.isnan(p_prop) else False,
            "ConvNeXt2-G2 MSE": same_mean,
            "Delta vs ConvNeXt2-G2": mean_v - same_mean,
            "Wilcoxon p (vs. ConvNeXt2-G2)": p_same,
            "Significant vs ConvNeXt2-G2": bool(p_same < 0.05) if not np.isnan(p_same) else False,
        })

    df = pd.DataFrame(rows)
    out = os.path.join(run_dir(cfg), f"convnext_ft_ablation_{region_key}.csv")
    df.to_csv(out, index=False)
    print(f"\n[{CONVNEXT_FT_LABEL}] {region_key}, Top-1 to Top-{max(TOP_K_LIST)}")
    print(df.to_string(index=False))
    print(f"Saved: {os.path.basename(out)}")

    # Seasonal results in the same per-k form. seasonal_generalization only
    # iterates over the shared dicts, so ConvNeXt-FT is computed here.
    from .seasonal_generalization import evaluate_by_season
    srows = []
    for k in TOP_K_LIST:
        res = evaluate_by_season(top_idx, query_eval_dts, db_eval_dts, k)
        row = {"Region": region_key, "Model": CONVNEXT_FT_LABEL, "Top-k": k}
        for season, (mse, n) in res.items():
            row[f"{season} MSE"] = mse
            row[f"{season} N"] = n
        srows.append(row)
    sdf = pd.DataFrame(srows)
    sout = os.path.join(run_dir(cfg), f"convnext_ft_seasonal_{region_key}.csv")
    sdf.to_csv(sout, index=False)
    print(f"Saved: {os.path.basename(sout)}")

    np.savez_compressed(
        os.path.join(run_dir(cfg), f"convnext_ft_topidx_{region_key}.npz"),
        top_idx=np.asarray(top_idx),
    )
    return df, sdf


# =========================================================
# Per-model retrieval results for a given query
# =========================================================
def inspect_query_comparison(query_position, region_key="south_korea", top_k_show=5,
                              models=None, cfg=CONFIG):
    """query_position: position in target_indices (0 to N_QUERY-1).
       For a raw HDF5 index, convert it first with raw_index_to_query_position().
       models=None compares all models in FINAL_REPORT_TOPIDX[region_key]."""
    from ..eval_mse_retrieval import FINAL_REPORT_TOPIDX

    topidx_dict = FINAL_REPORT_TOPIDX.get(region_key, {})
    if not topidx_dict:
        print(f"FINAL_REPORT_TOPIDX['{region_key}'] is empty. Run eval_mse_retrieval first.")
        return
    models = models or list(topidx_dict.keys())

    raw_query_idx = target_indices[query_position]
    query_dt = query_eval_dts[query_position]
    print(f"Query position: {query_position}  |  Raw HDF5 index: {raw_query_idx}  |  datetime: {query_dt}")

    box = REGIONS[region_key]["box"]  # box in pixels (ViT layout), reusable for any model
    bx, by, bw, bh = box

    with h5py.File(cfg["paths"]["h5"], "r") as f:
        dset = f['images']
        query_img = dset[raw_query_idx - 1]
        if query_img.shape[0] == 3: query_img = query_img.transpose(1, 2, 0)

        fig, axes = plt.subplots(len(models), top_k_show + 1, figsize=(4 * (top_k_show + 1), 4 * len(models)))
        if len(models) == 1:
            axes = axes[None, :]

        for row, model_name in enumerate(models):
            if model_name not in topidx_dict:
                print(f"No results for '{model_name}', skipping")
                continue
            top_idx_row = topidx_dict[model_name][query_position][:top_k_show]

            axes[row, 0].imshow(query_img)
            axes[row, 0].add_patch(patches.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='red', facecolor='none'))
            axes[row, 0].set_title(f"Query (t)\n{query_dt}\n[{model_name}]", fontsize=10, fontweight='bold')
            axes[row, 0].axis('off')

            for col, db_rank in enumerate(top_idx_row):
                raw_db_idx = db_indices[db_rank]  # top_idx is a position in db_indices -> convert to the raw HDF5 index
                found_dt = db_eval_dts[db_rank]
                found_img = dset[raw_db_idx]
                if found_img.shape[0] == 3: found_img = found_img.transpose(1, 2, 0)

                title_col, mse_str = 'black', "N/A"
                if query_dt in weather_map and found_dt in weather_map:
                    val = mean_squared_error(weather_map[query_dt], weather_map[found_dt])
                    mse_str = f"{val:.4f}"
                    title_col = 'blue' if val < 0.5 else ('red' if val > 2.0 else 'black')

                ax = axes[row, col + 1]
                ax.imshow(found_img)
                ax.add_patch(patches.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='red', facecolor='none'))
                ax.set_title(f"Top-{col+1}\n{found_dt}\nMSE: {mse_str}", fontsize=10, color=title_col, fontweight='bold')
                ax.axis('off')

        plt.suptitle(f"Query position {query_position} ({query_dt}), Region: {region_key}", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir(cfg), f"query_comparison_pos{query_position}.png"), dpi=200, bbox_inches='tight')
        plt.show()


def raw_index_to_query_position(raw_hdf5_index):
    """Return the position of a raw HDF5 index (0-based) among the 500 sampled queries,
    or None if it is not in target_indices (i.e. it was not sampled as a query)."""
    matches = np.where(target_indices == raw_hdf5_index)[0]
    return int(matches[0]) if len(matches) > 0 else None


# =========================================================
# Ad hoc retrieval comparison for an arbitrary index (DB embeddings are computed once per model and cached)
# =========================================================
_DB_PATCH_CACHE = {}


def get_or_build_db_patches(model_name, region_key="south_korea", cfg=CONFIG):
    from ..eval_setup import SPLIT_IDX
    from ..regions import region_tensor

    key = (model_name, region_key, cfg["patch_emb_size"])
    if key in _DB_PATCH_CACHE:
        return _DB_PATCH_CACHE[key]

    if model_name == "Proposed":
        model = load_trained_model(cfg=cfg, device=device)
        curr_indices = region_tensor(region_key, device)
    elif model_name == "ConvNeXt-FT":
        ckpt = os.path.join(run_dir(cfg), "convnext_finetuned_baseline.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"{ckpt} not found. Run train_finetuned_convnext_baseline() first.")
        model = ConvNeXtSpatialEncoder(cfg["patch_emb_size"]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        curr_indices = torch.tensor(REGIONS_CONVNEXT[region_key]["indices"], dtype=torch.long, device=device)
    else:
        raise ValueError(f"Unknown model_name: {model_name} (only 'Proposed' and 'ConvNeXt-FT' are supported)")

    db_patches = []
    with h5py.File(cfg["paths"]["h5"], "r") as f, torch.no_grad():
        dset = f['images']
        for start_b in tqdm(range(cfg["seq_len"], SPLIT_IDX, cfg["batch_size"]), desc=f"[{model_name}] DB caching"):
            end_b = min(start_b + cfg["batch_size"], SPLIT_IDX)
            img_t = normalize_image(torch.tensor(dset[start_b:end_b], dtype=torch.float32)).to(device)
            db_patches.append(model.forward_spatial(img_t).index_select(1, curr_indices).cpu())
    db_patches = torch.cat(db_patches, dim=0).to(device)
    _DB_PATCH_CACHE[key] = (model, db_patches, curr_indices)
    print(f"[{model_name} / {region_key}] cached DB embeddings ({len(db_patches)})")
    return _DB_PATCH_CACHE[key]


def inspect_raw_index(query_img_idx, region_key="south_korea", top_k_show=5,
                       models=("Proposed", "ConvNeXt-FT"), cfg=CONFIG):
    """query_img_idx: any raw HDF5 index in [CONFIG['seq_len'], SPLIT_IDX).
       Works regardless of whether it is in the 500-query sample (target_indices)."""
    from ..eval_setup import SPLIT_IDX

    assert cfg["seq_len"] <= query_img_idx < SPLIT_IDX, (
        f"query_img_idx must be in [{cfg['seq_len']}, {SPLIT_IDX}) (the encodable DB range)."
    )
    box = REGIONS[region_key]["box"]
    bx, by, bw, bh = box

    with h5py.File(cfg["paths"]["h5"], "r") as f:
        dset = f['images']
        query_img = dset[query_img_idx]
        if query_img.shape[0] == 3: query_img = query_img.transpose(1, 2, 0)
        dt_raw = f['datetimes'][query_img_idx]
        query_dt = dt_raw.decode() if isinstance(dt_raw, bytes) else str(dt_raw)

    print(f"Raw HDF5 index: {query_img_idx}  |  datetime: {query_dt}")

    fig, axes = plt.subplots(len(models), top_k_show + 1, figsize=(4 * (top_k_show + 1), 4 * len(models)))
    if len(models) == 1:
        axes = axes[None, :]

    for row, model_name in enumerate(models):
        model, db_patches, curr_indices = get_or_build_db_patches(model_name, region_key, cfg)

        with h5py.File(cfg["paths"]["h5"], "r") as f, torch.no_grad():
            dset = f['images']
            img_t = normalize_image(torch.tensor(dset[query_img_idx], dtype=torch.float32).unsqueeze(0)).to(device)
            q_emb = F.normalize(model.forward_spatial(img_t).index_select(1, curr_indices), p=2, dim=2)

        db_norm = F.normalize(db_patches, p=2, dim=2)
        sims = (db_norm * q_emb).sum(dim=2).mean(dim=1)  # (N_db,)
        top_scores, top_ranks = torch.topk(sims, k=top_k_show)
        top_ranks = top_ranks.cpu().numpy()

        axes[row, 0].imshow(query_img)
        axes[row, 0].add_patch(patches.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='red', facecolor='none'))
        axes[row, 0].set_title(f"Query (t)\n{query_dt}\n[{model_name}]", fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')

        with h5py.File(cfg["paths"]["h5"], "r") as f:
            dset = f['images']
            for col, rank in enumerate(top_ranks):
                raw_db_idx = db_indices[rank]
                found_dt = db_eval_dts[rank]
                found_img = dset[raw_db_idx]
                if found_img.shape[0] == 3: found_img = found_img.transpose(1, 2, 0)

                title_col, mse_str = 'black', "N/A"
                if query_dt in weather_map and found_dt in weather_map:
                    val = mean_squared_error(weather_map[query_dt], weather_map[found_dt])
                    mse_str = f"{val:.4f}"
                    title_col = 'blue' if val < 0.5 else ('red' if val > 2.0 else 'black')

                ax = axes[row, col + 1]
                ax.imshow(found_img)
                ax.add_patch(patches.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='red', facecolor='none'))
                ax.set_title(f"Top-{col+1}\n{found_dt}\nSim:{top_scores[col]:.3f} MSE:{mse_str}",
                              fontsize=9, color=title_col, fontweight='bold')
                ax.axis('off')

    plt.suptitle(f"Raw index {query_img_idx} ({query_dt}), Region: {region_key}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir(cfg), f"raw_index_comparison_{query_img_idx}.png"), dpi=200, bbox_inches='tight')
    plt.show()


def main():
    run_pixel_nn_baseline()

    # NOTE: this baseline (torchvision ConvNeXt-Tiny) is separate from the
    # timm-based ConvNeXt2 in baselines/.
    train_finetuned_convnext_baseline()

    evaluate_convnext_finetuned()

    # Usage:
    # 1) Pick a position among the 500 queries.
    # ConvNeXt-FT is not merged into FINAL_REPORT_TOPIDX, so it cannot be
    # selected by name here. Its rankings are in convnext_ft_topidx_<region>.npz;
    # for visual comparison use inspect_raw_index(models=("Proposed", "ConvNeXt-FT")),
    # which loads the checkpoint directly (see below).
    inspect_query_comparison(query_position=0, region_key="south_korea")

    # 2) Given a raw HDF5 index:
    # pos = raw_index_to_query_position(12345)
    # if pos is not None:
    #     inspect_query_comparison(query_position=pos)

    # Example: replace with any index (the first call is slow due to DB caching)
    inspect_raw_index(query_img_idx=1000, region_key="south_korea",
                       models=("Proposed", "ConvNeXt-FT"))


if __name__ == "__main__":
    main()
