"""Shared evaluation infrastructure: the query/DB split and retrieval metrics.

Every evaluation module (MSE retrieval, Table 1 image quality, the D/PE
ablations, and the baselines/ Group 1/2 comparisons) reuses the exact same
query/DB split and metric functions defined here instead of re-deriving them.

calculate_patch_mse_metrics ranks by mean per-patch cosine (paper Eq. 9) and is
used for RaP-Weather; calculate_mse_metrics ranks by global cosine between
single vectors and is used for the baselines.

Importing this module reproduces the deterministic (seeded) query sample and
DB/candidate index split; it does not do any GPU work.
"""
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from scipy import stats

from .config import CONFIG
from .models import get_weather_map_and_scaler

weather_map, weather_scaler = get_weather_map_and_scaler(CONFIG)

with h5py.File(CONFIG["paths"]["h5"], "r") as f:
    datetimes_full = f['datetimes'][:].astype(str)
    total_samples = len(datetimes_full)

SPLIT_IDX = int(total_samples * CONFIG["train_val_split"])
db_indices = np.arange(CONFIG["seq_len"], SPLIT_IDX)
candidate_indices = np.arange(SPLIT_IDX + CONFIG["seq_len"], total_samples)
rng = np.random.RandomState(CONFIG["seed"])
if CONFIG["n_query"] is None or CONFIG["n_query"] >= len(candidate_indices):
    # Complete test set, in natural temporal order: no subsample, so no seed
    # dependence and no researcher degree of freedom in which queries are scored.
    target_indices = candidate_indices.copy()
else:
    target_indices = rng.choice(candidate_indices, CONFIG["n_query"], replace=False)
N_QUERY_EFFECTIVE = len(target_indices)

db_eval_dts = [datetimes_full[idx] for idx in db_indices]
query_eval_dts = [datetimes_full[idx - 1] for idx in target_indices]
_scope = "FULL test set" if N_QUERY_EFFECTIVE == len(candidate_indices) else "subsample"
print(f"Query samples: {N_QUERY_EFFECTIVE} ({_scope} of {len(candidate_indices)}) "
      f"| DB size: {len(db_indices)}")

TOP_K_LIST = CONFIG["top_k_list"]


def calculate_patch_mse_metrics(query_patches, db_patches, target_dts, db_dts):
    q_norm = F.normalize(query_patches, p=2, dim=2)
    db_norm = F.normalize(db_patches, p=2, dim=2)
    P = query_patches.shape[1]
    sim_matrix = torch.einsum('ipd, jpd -> ij', q_norm, db_norm) / P
    _, top_indices = torch.topk(sim_matrix, k=max(TOP_K_LIST), dim=1)
    top_indices = top_indices.cpu().numpy()
    results = {k: [] for k in TOP_K_LIST}
    for i, t_dt in enumerate(target_dts):
        if t_dt not in weather_map: continue
        target_w = weather_map[t_dt]
        found_idxs = top_indices[i]
        for k in TOP_K_LIST:
            mses = [mean_squared_error(target_w, weather_map[db_dts[di]]) for di in found_idxs[:k] if db_dts[di] in weather_map]
            if mses: results[k].append(np.mean(mses))
    mean_results = {k: (np.mean(v) if v else np.nan) for k, v in results.items()}
    return mean_results, results, top_indices  # (mean, raw-per-query, ranked-indices)


def calculate_mse_metrics(query_vecs, db_vecs, target_dts, db_dts):
    q_norm = F.normalize(query_vecs, p=2, dim=1)
    db_norm = F.normalize(db_vecs, p=2, dim=1)
    sim_matrix = torch.mm(q_norm, db_norm.T)
    _, top_indices = torch.topk(sim_matrix, k=max(TOP_K_LIST), dim=1)
    top_indices = top_indices.cpu().numpy()
    results = {k: [] for k in TOP_K_LIST}
    for i, t_dt in enumerate(target_dts):
        if t_dt not in weather_map: continue
        target_w = weather_map[t_dt]
        found_idxs = top_indices[i]
        for k in TOP_K_LIST:
            mses = [mean_squared_error(target_w, weather_map[db_dts[di]]) for di in found_idxs[:k] if db_dts[di] in weather_map]
            if mses: results[k].append(np.mean(mses))
    mean_results = {k: (np.mean(v) if v else np.nan) for k, v in results.items()}
    return mean_results, results, top_indices


# Precision@k / Recall@k / MAP are not provided. They need a relevance label
# this dataset does not have; a stand-in such as "closest 10% of the DB in
# weather space" is an assumption, not ground truth, so the numbers would not be
# interpretable as retrieval quality. Weather-MSE over the retrieved set
# (calculate_*_mse_metrics above) is the metric this data can support.


def bootstrap_ci(values, n_boot=2000, ci=95, seed=42):
    values = np.asarray(values, dtype=np.float64)
    rng_b = np.random.RandomState(seed)
    boot_means = np.array([rng_b.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    return values.mean(), np.percentile(boot_means, (100-ci)/2), np.percentile(boot_means, 100-(100-ci)/2)


def paired_significance_report(raw_dict, baseline_name="Proposed"):
    """Bootstrap 95% CI + paired Wilcoxon signed-rank test vs. baseline_name,
    per Top-k. Pairing is valid: every model sees the identical
    (query, target-weather) pairs, in the same order."""
    rows, base_per_k = [], raw_dict.get(baseline_name, {})
    for model_name, per_k in raw_dict.items():
        for k, vals in per_k.items():
            vals = np.asarray(vals, dtype=np.float64)
            if len(vals) == 0: continue
            mean_v, lo, hi = bootstrap_ci(vals)
            row = {"Model": model_name, "Top-k": k, "Mean MSE": mean_v, "95% CI Low": lo, "95% CI High": hi, "N": len(vals)}
            if model_name != baseline_name and k in base_per_k:
                base_vals = np.asarray(base_per_k[k], dtype=np.float64)
                if len(base_vals) == len(vals) and len(vals) > 1:
                    try:
                        _, p = stats.wilcoxon(base_vals, vals)
                    except ValueError:
                        p = np.nan
                    row["Wilcoxon p (vs. Proposed)"] = p
                    row["Significant (p<0.05)"] = bool(p < 0.05) if not np.isnan(p) else False
            rows.append(row)
    return pd.DataFrame(rows)
