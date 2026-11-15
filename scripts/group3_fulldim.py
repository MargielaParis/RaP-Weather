"""Paper "Group 2", full dimension: backbone fine-tuned on whole images, region cropped at query time.

Code name "group3" corresponds to the paper's "Group 2"; code "group2" is the
paper's per-region encoders. This full-dimension variant is the one used in the
final paper; group3_wholetrain_crop.py is the D=8 projection variant.

Conditions and structure are the same as group3_wholetrain_crop.py. Only the model
differs: instead of the compressed WholeRegionPredictor (proj to D=8) it uses the
full-dimension group2_fulldim_ablation.FullDimRegionPredictor (backbone native
dimension).

Evaluation conditions are unchanged:
    Proposed / G3 resize / G3 pad-mean / G3 pad-zero / G2 region-trained

The full-dimension model has no proj, so the pad path is simpler than in the D=8
variant (the backbone output is the embedding). Crop transforms, sorted h5 reads
and significance tests reuse the same functions.

Run from the repo root: python scripts/group3_fulldim.py
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from tqdm import tqdm

from rap_weather.config import CONFIG, device, run_dir
from rap_weather.models import load_trained_model, normalize_image
from rap_weather.regions import REGIONS, region_tensor
from rap_weather.eval_setup import (
    bootstrap_ci, calculate_mse_metrics, calculate_patch_mse_metrics,
    db_eval_dts, db_indices, query_eval_dts, target_indices,
)
from rap_weather.baselines.backbone_registry import (
    BACKBONE_SPECS, crop_and_pad, crop_and_resize,
)
from group2_fulldim_ablation import FullDimRegionPredictor, ckpt_path

PAD_MODES = ("mean", "zero")
REGIONS_TO_RUN = ["south_korea"]
SELFCHECK_REGION = "total_patches"
TRAINED_ON = "total_patches"


def _load(backbone_name, ckpt_region, out_root):
    ckpt = ckpt_path(out_root, backbone_name, ckpt_region)
    if not os.path.exists(ckpt):
        print(f"  checkpoint not found, skipping: {ckpt}", flush=True)
        return None
    m = FullDimRegionPredictor(backbone_name, seq_len=CONFIG["seq_len"]).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m


def _encode(model, box, indices, desc, pad_mode=None):
    """h5py fancy indexing requires increasing indices -> sort, then restore the order."""
    indices = np.asarray(indices)
    order = np.argsort(indices)
    srt = indices[order]
    out = []
    with h5py.File(CONFIG["paths"]["h5"], "r") as f, torch.no_grad():
        dset = f["images"]
        for s in tqdm(range(0, len(srt), CONFIG["batch_size"]), desc=desc, leave=False):
            b = srt[s:s + CONFIG["batch_size"]]
            imgs = normalize_image(torch.tensor(dset[list(b)], dtype=torch.float32)).to(device)
            crop = (crop_and_resize(imgs, box, model.input_size) if pad_mode is None
                    else crop_and_pad(imgs, box, model.input_size, pad_mode))
            out.append(model.backbone(crop).cpu())
    return torch.cat(out, dim=0)[np.argsort(order)]


def eval_backbone(backbone_name, ckpt_region, eval_region, out_root, pad_mode=None):
    m = _load(backbone_name, ckpt_region, out_root)
    if m is None:
        return None, None
    box = REGIONS[eval_region]["box"]
    tag = f"[{backbone_name}|{ckpt_region}->{eval_region}" + (f"|pad={pad_mode}]" if pad_mode else "]")
    db = _encode(m, box, db_indices, f"{tag} DB", pad_mode)
    q = _encode(m, box, target_indices - 1, f"{tag} Q", pad_mode)
    mean_res, raw_res, _ = calculate_mse_metrics(q.to(device), db.to(device), query_eval_dts, db_eval_dts)
    del m, db, q
    torch.cuda.empty_cache()
    return mean_res, raw_res


def eval_proposed(region_key):
    """Proposed: encode the whole image once -> select the region's patch indices (no crop)."""
    model = load_trained_model(cfg=CONFIG, device=device)
    model.eval()
    idx = region_tensor(region_key, device)

    def enc(indices, desc):
        indices = np.asarray(indices)
        order = np.argsort(indices)
        srt = indices[order]
        out = []
        with h5py.File(CONFIG["paths"]["h5"], "r") as f, torch.no_grad():
            dset = f["images"]
            for s in tqdm(range(0, len(srt), CONFIG["batch_size"]), desc=desc, leave=False):
                b = srt[s:s + CONFIG["batch_size"]]
                imgs = normalize_image(torch.tensor(dset[list(b)], dtype=torch.float32)).to(device)
                out.append(model.forward_spatial(imgs).index_select(1, idx).cpu())
        return torch.cat(out, dim=0)[np.argsort(order)]

    db = enc(db_indices, "[Proposed] DB")
    q = enc(target_indices - 1, "[Proposed] Q")
    mean_res, raw_res, _ = calculate_patch_mse_metrics(q.to(device), db.to(device), query_eval_dts, db_eval_dts)
    del model, db, q
    torch.cuda.empty_cache()
    return mean_res, raw_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=REGIONS_TO_RUN)
    ap.add_argument("--skip-selfcheck", action="store_true")
    args = ap.parse_args()

    d = run_dir(CONFIG)
    print(f"queries: {len(target_indices)} | DB: {len(db_indices)} | regions: {args.regions}", flush=True)

    for region in args.regions:
        means, raws = {}, {}
        print(f"\n=== [{region}] Proposed (no crop, patch selection) ===", flush=True)
        means["Proposed"], raws["Proposed"] = eval_proposed(region)

        for bb in BACKBONE_SPECS:
            for label, ck, pm in [
                (f"{bb} (G3: whole-trained, crop@query)", TRAINED_ON, None),
                (f"{bb} (G3-pad/mean: whole-trained, crop@query)", TRAINED_ON, "mean"),
                (f"{bb} (G3-pad/zero: whole-trained, crop@query)", TRAINED_ON, "zero"),
                (f"{bb} (G2: region-trained)", region, None),
            ]:
                print(f"\n=== [{region}] {label} ===", flush=True)
                m, r = eval_backbone(bb, ck, region, d, pm)
                if m is not None:
                    means[label], raws[label] = m, r
                    print(f"   Top-1 {m[1]:.4f} | Top-100 {m[100]:.4f}", flush=True)

        df = pd.DataFrame(means).T
        df.columns = [f"Top-{k}" for k in df.columns]
        out_csv = os.path.join(d, f"group3_fulldim_wholetrain_crop_{region}.csv")
        df.to_csv(out_csv)
        print(f"\n[{region}] mean weather-MSE\n{df.to_string()}", flush=True)

        rows, base = [], raws["Proposed"]
        for name, per_k in raws.items():
            for k, vals in per_k.items():
                v = np.asarray(vals, dtype=np.float64)
                if v.size == 0:
                    continue
                mean_v, lo, hi = bootstrap_ci(v)
                row = {"Model": name, "Top-k": k, "Mean MSE": mean_v,
                       "95% CI Low": lo, "95% CI High": hi,
                       "Median": float(np.median(v)), "N": len(v)}
                if name != "Proposed" and k in base:
                    b = np.asarray(base[k], dtype=np.float64)
                    if len(b) == len(v):
                        try:
                            row["Wilcoxon p (vs. Proposed)"] = wilcoxon(v, b).pvalue
                        except ValueError:
                            row["Wilcoxon p (vs. Proposed)"] = np.nan
                rows.append(row)
        sig = pd.DataFrame(rows)
        sig["Significant (p<0.05)"] = sig.get("Wilcoxon p (vs. Proposed)", np.nan) < 0.05
        sig.to_csv(os.path.join(d, f"group3_fulldim_significance_{region}.csv"), index=False)
        print(f"\n[{region}] Top-1 summary")
        print(sig[sig["Top-k"] == 1].to_string(index=False), flush=True)

        np.savez_compressed(
            os.path.join(d, f"group3_fulldim_perquery_raw_{region}.npz"),
            **{f"{n}||Top-{k}": np.asarray(v, dtype=np.float64)
               for n, pk in raws.items() for k, v in pk.items()},
        )

    if args.skip_selfcheck:
        print("\nselfcheck skipped.\ndone.", flush=True)
        return

    # ---- Sanity check: on total_patches the result must match group2_fulldim_total_patches.csv ----
    print(f"\n=== sanity check (ckpt={TRAINED_ON}, eval={SELFCHECK_REGION}) ===", flush=True)
    ref_path = os.path.join(d, f"group2_fulldim_{SELFCHECK_REGION}.csv")
    ref = pd.read_csv(ref_path, index_col=0) if os.path.exists(ref_path) else None
    for bb in BACKBONE_SPECS:
        m, _ = eval_backbone(bb, TRAINED_ON, SELFCHECK_REGION, d)
        if m is None or ref is None or bb not in ref.index:
            continue
        got, exp = m[1], float(ref.loc[bb, "Top-1"])
        ok = abs(got - exp) < 1e-9
        print(f"  {bb:10s} Top-1 got={got:.10f} expected={exp:.10f} -> "
              f"{'MATCH' if ok else 'MISMATCH'}", flush=True)
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
