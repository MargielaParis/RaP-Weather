"""Table 1 (LPIPS / PSNR / SSIM) scored against region-aware baselines rather
than the precomputed-pkl baselines.

eval_image_quality.py only iterates the models inside baseline_pkl, i.e.
whole-frame precomputed vectors that never crop to the query region. Here the
baselines are Group 1 (off-the-shelf, crop -> encode), the like-for-like
off-the-shelf baseline, and the whole-image-trained encoders cropped at query
time, the condition the paper's core claim rests on.

Naming: code "Group3" / topk_group3 is the paper's Group 2 (whole-image
trained, crop at query); code "Group2" / topk_group2 is the per-region encoder
of Appendix B.

Speed: eval_image_quality calls LPIPS once per (query, retrieved) pair, i.e.
5,143 x 100 separate GPU round-trips per model. Here the ROIs are preloaded
into RAM once and LPIPS runs batched over all 100 retrieved images of a query,
which keeps the 7 model x region combinations affordable.

Metric definitions are copied from eval_image_quality.calculate_metrics so the
numbers stay comparable with that table.

Run from the repo root: python scripts/table1_group13.py [--regions ...] [--fulldim]
"""
import argparse
import json
import os
import time

import h5py
import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from skimage.metrics import structural_similarity as ssim_func
from tqdm import tqdm

from rap_weather.config import CONFIG, device, run_dir
from rap_weather.models import load_trained_model, normalize_image
from rap_weather.regions import REGIONS, region_tensor
from rap_weather.eval_setup import (
    SPLIT_IDX, TOP_K_LIST, bootstrap_ci, db_indices, target_indices,
)
from rap_weather.baselines.backbone_registry import (
    BACKBONE_SPECS, crop_and_resize, get_frozen_backbone,
)
from rap_weather.baselines.group2_finetuned_baseline import (
    WholeRegionPredictor, group2_checkpoint_path,
)

MAX_K = max(TOP_K_LIST)
loss_fn_alex = lpips.LPIPS(net="alex").to(device)


# ----------------------------------------------------------------- ROI cache
def load_rois(region_key, cfg=CONFIG):
    """DB and query ROIs as uint8 in RAM, read once instead of ~514k random HDF5
    reads per model."""
    x, y, w, h = REGIONS[region_key]["box"]
    with h5py.File(cfg["paths"]["h5"], "r") as f:
        dset = f["images"]
        chw = dset.shape[1] == 3          # (N,3,H,W) vs (N,H,W,3)

        def grab(indices, desc, chunk=512):
            """db_indices and target_indices-1 are both contiguous ranges, so read
            them as slices; 20k single-image reads is an order of magnitude
            slower than the same bytes pulled in chunks."""
            indices = np.asarray(indices)
            lo, hi = int(indices[0]), int(indices[-1]) + 1
            contiguous = (hi - lo == len(indices)) and bool(np.all(np.diff(indices) == 1))
            out = np.empty((len(indices), h, w, 3), dtype=np.uint8)
            pos = 0
            for s in tqdm(range(lo, hi, chunk), desc=desc, leave=False):
                e = min(s + chunk, hi)
                blk = dset[s:e] if contiguous else dset[list(indices[pos:pos + (e - s)])]
                if chw:
                    blk = blk.transpose(0, 2, 3, 1)
                n = len(blk)
                out[pos:pos + n] = blk[:, y:y + h, x:x + w, :].astype(np.uint8)
                pos += n
            return out

        db = grab(db_indices, f"[{region_key}] DB ROIs")
        q = grab(target_indices - 1, f"[{region_key}] query ROIs")
    return db, q


# ------------------------------------------------------------------ encoders
def _encode_batched(fn, indices, cfg, desc):
    """h5py fancy-indexing needs ascending order; sort then undo."""
    indices = np.asarray(indices)
    order = np.argsort(indices)
    out = []
    with h5py.File(cfg["paths"]["h5"], "r") as f, torch.no_grad():
        dset = f["images"]
        srt = indices[order]
        for s in tqdm(range(0, len(srt), cfg["batch_size"]), desc=desc, leave=False):
            b = srt[s:s + cfg["batch_size"]]
            imgs = normalize_image(torch.tensor(dset[list(b)], dtype=torch.float32)).to(device)
            out.append(fn(imgs).cpu())
    return torch.cat(out, 0)[np.argsort(order)]


def topk_proposed(region_key, cfg=CONFIG):
    model = load_trained_model(cfg=cfg, device=device); model.eval()
    idx = region_tensor(region_key, device)
    enc = lambda im: model.forward_spatial(im).index_select(1, idx)
    db = _encode_batched(enc, db_indices, cfg, "[RaP] DB")
    q = _encode_batched(enc, target_indices - 1, cfg, "[RaP] Q")
    db_n = F.normalize(db.to(device), p=2, dim=2)
    q_n = F.normalize(q.to(device), p=2, dim=2)
    sims = torch.einsum("ipd,jpd->ij", q_n, db_n) / db_n.shape[1]
    top = torch.topk(sims, k=MAX_K, dim=1).indices.cpu().numpy()
    del model, db, q, db_n, q_n
    torch.cuda.empty_cache()
    return top


def topk_vector(enc_fn, tag, cfg=CONFIG):
    db = _encode_batched(enc_fn, db_indices, cfg, f"[{tag}] DB")
    q = _encode_batched(enc_fn, target_indices - 1, cfg, f"[{tag}] Q")
    db_n = F.normalize(db.to(device), p=2, dim=1)
    q_n = F.normalize(q.to(device), p=2, dim=1)
    top = torch.topk(q_n @ db_n.T, k=MAX_K, dim=1).indices.cpu().numpy()
    del db, q, db_n, q_n
    torch.cuda.empty_cache()
    return top


def topk_group1(backbone, region_key, cfg=CONFIG):
    model, in_size, _ = get_frozen_backbone(backbone)
    box = REGIONS[region_key]["box"]
    return topk_vector(lambda im: model(crop_and_resize(im, box, in_size)),
                       f"G1-{backbone}", cfg)


# --fulldim: swap the compressed (proj -> D=8) code Group 2/3 encoder for the
# native-dimension one. Only the encoder and its checkpoint path change; the
# ranking and LPIPS/PSNR/SSIM scoring paths below stay shared so the two
# protocols remain numerically comparable.
FULLDIM = False


def _load_g23(backbone, ckpt_region, cfg):
    """Returns (model, forward_fn) or None when the checkpoint is missing."""
    if FULLDIM:
        from group2_fulldim_ablation import FullDimRegionPredictor, ckpt_path
        ckpt = ckpt_path(run_dir(cfg), backbone, ckpt_region)
        if not os.path.exists(ckpt):
            return None
        m = FullDimRegionPredictor(backbone, seq_len=cfg["seq_len"]).to(device)
    else:
        ckpt = group2_checkpoint_path(backbone, ckpt_region, cfg)
        if not os.path.exists(ckpt):
            return None
        m = WholeRegionPredictor(backbone, embed_dim=cfg["patch_emb_size"],
                                 seq_len=cfg["seq_len"]).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()
    return m


def topk_group2(backbone, region_key, cfg=CONFIG):
    """Code Group 2 (paper per-region encoder, Appendix B): the checkpoint trained
    on this region, evaluated on it. Same weights as the weather-MSE table."""
    m = _load_g23(backbone, region_key, cfg)
    if m is None:
        return None
    box = REGIONS[region_key]["box"]
    top = topk_vector(lambda im: m.forward_spatial(im, box), f"G2-{backbone}", cfg)
    del m
    torch.cuda.empty_cache()
    return top


def topk_group3(backbone, region_key, cfg=CONFIG):
    """Code Group 3 (paper Group 2): the whole-image-trained checkpoint, cropped
    at query time."""
    m = _load_g23(backbone, "total_patches", cfg)
    if m is None:
        return None
    box = REGIONS[region_key]["box"]
    top = topk_vector(lambda im: m.forward_spatial(im, box), f"G3-{backbone}", cfg)
    del m
    torch.cuda.empty_cache()
    return top


# ------------------------------------------------------------------- scoring
def score(top_idx, db_rois, q_rois, tag):
    """LPIPS batched per query; PSNR vectorised; SSIM per pair (skimage, to keep
    the number identical to the existing table)."""
    min_side = min(q_rois.shape[1], q_rois.shape[2])
    win = min(7, min_side) if min_side % 2 else min(7, min_side - 1)
    acc = {m: {k: [] for k in TOP_K_LIST} for m in ("LPIPS", "PSNR", "SSIM")}

    for i in tqdm(range(len(q_rois)), desc=f"[{tag}] scoring", leave=False):
        t = q_rois[i]
        r = db_rois[top_idx[i]]                                   # (MAX_K,h,w,3)

        diff = (r.astype(np.float64) - t.astype(np.float64)) ** 2
        mse = diff.reshape(len(r), -1).mean(1)
        psnr = np.where(mse == 0, 100.0, 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-12)))

        tt = torch.from_numpy(t.transpose(2, 0, 1)).float().div(255).mul(2).sub(1)
        rr = torch.from_numpy(r.transpose(0, 3, 1, 2)).float().div(255).mul(2).sub(1)
        with torch.no_grad():
            lp = loss_fn_alex(tt.unsqueeze(0).expand_as(rr).to(device),
                              rr.to(device)).flatten().cpu().numpy()

        ss = np.array([ssim_func(t, rj, data_range=255, channel_axis=2, win_size=win)
                       for rj in r])

        for k in TOP_K_LIST:
            acc["LPIPS"][k].append(lp[:k].mean())
            acc["PSNR"][k].append(psnr[:k].mean())
            acc["SSIM"][k].append(ss[:k].mean())
    return acc


def suffix_of(args):
    return (f"_limit{args.limit}" if args.limit else "") + ("_fulldim" if args.fulldim else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=["south_korea", "total_patches"])
    ap.add_argument("--limit", type=int, default=None, help="score only the first N queries (benchmark)")
    ap.add_argument("--append", action="store_true",
                    help="reuse per-query results already in the npz; only score models missing from it")
    ap.add_argument("--fulldim", action="store_true",
                    help="score the native-dimension Group 2/3 encoders instead of the compressed (D=8) ones")
    args = ap.parse_args()
    global FULLDIM
    FULLDIM = args.fulldim
    if FULLDIM:
        print("[--fulldim] native-dimension Group 2/3 encoders", flush=True)

    out_dir = run_dir(CONFIG)
    rows, sig_rows = [], []
    for region in args.regions:
        print(f"\n{'='*60}\nregion: {region}\n{'='*60}", flush=True)
        db_rois, q_rois = load_rois(region)
        if args.limit:
            q_rois = q_rois[:args.limit]

        raw = {}
        npz_path = os.path.join(out_dir, f"table1_group13_perquery_{region}{suffix_of(args)}.npz")
        if args.append and os.path.exists(npz_path):
            z = np.load(npz_path)
            for key in z.files:
                n, m, k = key.split("||")
                raw.setdefault(n, {}).setdefault(m, {})[int(k)] = list(z[key])
            print(f"  loaded {len(raw)} models from {os.path.basename(npz_path)}", flush=True)
        models = {"RaP-Weather": lambda r=region: topk_proposed(r)}
        for bb in BACKBONE_SPECS:
            models[f"{bb} (Group1, off-the-shelf)"] = lambda b=bb, r=region: topk_group1(b, r)
        # The whole-image-trained checkpoint fed a region crop. At total_patches
        # the crop is the whole frame, so the condition reduces to the plain
        # fine-tuned encoder and is labelled as such.
        g_lab = ("Group3, whole-trained + crop" if region != "total_patches"
                 else "Group2, predictive fine-tuned")
        for bb in BACKBONE_SPECS:
            models[f"{bb} ({g_lab})"] = lambda b=bb, r=region: topk_group3(b, r)

        # Per-region encoders (code Group 2), so this table uses the same
        # baselines as the weather-MSE table. ConvNeXt-FT is deliberately left out
        # of the visual table. At total_patches the region-trained checkpoint is
        # the whole-trained one, already covered by the block above.
        if region != "total_patches":
            for bb in BACKBONE_SPECS:
                models[f"{bb} (Group2, predictive fine-tuned)"] = lambda b=bb, r=region: topk_group2(b, r)

        for name, fn in models.items():
            if name in raw:
                print(f"  reuse {name}", flush=True)
                acc = raw[name]
                row = {"Scenario": region, "Model": name, "N": len(acc["LPIPS"][TOP_K_LIST[0]])}
                for m in ("LPIPS", "PSNR", "SSIM"):
                    for k in TOP_K_LIST:
                        row[f"{m}@{k}"] = float(np.mean(acc[m][k]))
                rows.append(row)
                continue
            t0 = time.time()
            top = fn()
            if top is None:
                print(f"  skip {name} (no checkpoint)", flush=True)
                continue
            acc = score(top[:len(q_rois)], db_rois, q_rois, name)
            raw[name] = acc
            row = {"Scenario": region, "Model": name, "N": len(q_rois)}
            for m in ("LPIPS", "PSNR", "SSIM"):
                for k in TOP_K_LIST:
                    row[f"{m}@{k}"] = float(np.mean(acc[m][k]))
            rows.append(row)
            print(f"  {name:<42} LPIPS@1 {row['LPIPS@1']:.4f} | "
                  f"PSNR@1 {row['PSNR@1']:.2f} | SSIM@1 {row['SSIM@1']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        # --- significance: bootstrap 95% CI + paired Wilcoxon vs RaP-Weather ---
        # Pairing is valid because every model scores the identical query list in
        # the same order. Same construction as eval_image_quality.run_table1_significance
        # so the two tables can sit side by side.
        base = raw.get("RaP-Weather")
        for name, acc in raw.items():
            for metric in ("LPIPS", "PSNR", "SSIM"):
                for k in TOP_K_LIST:
                    v = np.asarray(acc[metric][k], dtype=np.float64)
                    if v.size == 0:
                        continue
                    mean_v, lo, hi = bootstrap_ci(v)
                    r = {"Scenario": region, "Metric": metric, "Model": name, "Top-k": k,
                         "Mean": mean_v, "95% CI Low": lo, "95% CI High": hi, "N": len(v)}
                    if base is not None and name != "RaP-Weather":
                        b = np.asarray(base[metric][k], dtype=np.float64)
                        if len(b) == len(v) and len(v) > 1:
                            try:
                                p_ = stats.wilcoxon(b, v).pvalue
                            except ValueError:
                                p_ = np.nan
                            r["Wilcoxon p (vs. RaP-Weather)"] = p_
                            r["Significant (p<0.05)"] = bool(p_ < 0.05) if not np.isnan(p_) else False
                            # lower is better for LPIPS, higher for PSNR/SSIM
                            better = (v.mean() < b.mean()) if metric == "LPIPS" else (v.mean() > b.mean())
                            r["Better than RaP"] = bool(better)
                    sig_rows.append(r)

        np.savez_compressed(
            os.path.join(out_dir, f"table1_group13_perquery_{region}{suffix_of(args)}.npz"),
            **{f"{n}||{m}||{k}": np.asarray(a[m][k], dtype=np.float64)
               for n, a in raw.items() for m in a for k in a[m]},
        )
        del db_rois, q_rois, raw

    suffix = suffix_of(args)
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, f"table1_group13{suffix}.csv")
    df.to_csv(path, index=False)
    sig = pd.DataFrame(sig_rows)
    sig_path = os.path.join(out_dir, f"table1_group13_significance{suffix}.csv")
    sig.to_csv(sig_path, index=False)

    print(f"\nsaved: {path}\nsaved: {sig_path}")
    print(df[["Scenario", "Model", "LPIPS@1", "PSNR@1", "SSIM@1"]].to_string(index=False))
    q = sig[(sig["Top-k"] == 1) & (sig.Model != "RaP-Weather")]
    if len(q):
        print("\nTop-1 significance vs RaP-Weather:")
        print(q[["Scenario", "Metric", "Model", "Mean",
                 "Wilcoxon p (vs. RaP-Weather)", "Significant (p<0.05)",
                 "Better than RaP"]].to_string(index=False))


if __name__ == "__main__":
    main()
