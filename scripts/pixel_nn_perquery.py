"""Pixel-NN baseline: per-query weather-MSE results.

studies/baseline_comparison.calculate_pixel_space_mse_metrics returns per-query
values but does not save them, so they cannot be used in the unified_main_*
significance tests. This script reruns the same computation and saves the
per-query values to an npz. Importing that module triggers
eval_mse_retrieval.FINAL_REPORT_RAW (a full re-evaluation), so only the function
is copied here; its logic and order of operations must stay identical to
studies/baseline_comparison.py::calculate_pixel_space_mse_metrics.

Check: the Top-1..100 means must match the reported pixel_nn_baseline.csv (<1e-9).

Run from the repo root: python scripts/pixel_nn_perquery.py
Output: <run_dir>/pixel_nn_perquery_south_korea.npz (keys "Pixel-NN||Top-{k}")
"""
import os

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from rap_weather.config import CONFIG, run_dir
from rap_weather.regions import REGIONS
from rap_weather.eval_setup import (
    TOP_K_LIST, db_eval_dts, db_indices, query_eval_dts, target_indices, weather_map,
)


def calculate_pixel_space_mse_metrics(query_indices, db_index_range, target_dts, db_dts,
                                        region_key="south_korea", downsample_to=32, cfg=CONFIG):
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


def main():
    d = run_dir(CONFIG)
    mean_r, raw = calculate_pixel_space_mse_metrics(
        target_indices, db_indices, query_eval_dts, db_eval_dts, region_key="south_korea")

    ref = pd.read_csv(os.path.join(d, "pixel_nn_baseline.csv")).iloc[0]
    print("\n=== check against reported values ===")
    ok = True
    for k in TOP_K_LIST:
        diff = abs(mean_r[k] - ref[f"Top-{k}"])
        ok &= diff < 1e-9
        print(f"Top-{k:>3}: recomputed {mean_r[k]:.10f}  reported {ref[f'Top-{k}']:.10f}  |diff| {diff:.2e}")
    if not ok:
        raise SystemExit("mismatch with reported values; per-query results not saved")

    out = os.path.join(d, "pixel_nn_perquery_south_korea.npz")
    np.savez_compressed(out, **{f"Pixel-NN||Top-{k}": np.asarray(raw[k], dtype=np.float64)
                                for k in TOP_K_LIST})
    print(f"matches reported values; saved: {out}  (N={len(raw[TOP_K_LIST[0]])})")


if __name__ == "__main__":
    main()
