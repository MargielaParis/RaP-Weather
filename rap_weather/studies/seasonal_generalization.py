"""Seasonal and regional generalization analysis.

Addresses whether retrieval quality generalizes across seasons and regions.

Reuses FINAL_REPORT_TOPIDX (ranking indices over all queries, before filtering)
populated by eval_mse_retrieval.run_evaluation() and splits Top-k weather MSE by
season. Regional generalization is already covered because that evaluation
loop runs over all four regions in REGIONS_TO_EVAL.

Season labels such as "JJA (summer/monsoon)" also form the CSV column names
(f"{season} MSE").
"""
import os

import numpy as np
import pandas as pd

from ..config import CONFIG, run_dir
from ..eval_setup import TOP_K_LIST, weather_map, db_eval_dts, query_eval_dts
from ..eval_mse_retrieval import FINAL_REPORT_TOPIDX


def month_to_season(month):
    if month in (12, 1, 2): return "DJF (winter)"
    if month in (3, 4, 5): return "MAM (spring)"
    if month in (6, 7, 8): return "JJA (summer/monsoon)"
    return "SON (autumn)"


def parse_month(dt_str):
    """Assumes dt_str is YYYYMMDDHHMM (the strptime format used elsewhere).
    Adjust only here if the format differs."""
    try:
        return int(dt_str[4:6])
    except Exception:
        return None


SEASONS = ("DJF (winter)", "MAM (spring)", "JJA (summer/monsoon)", "SON (autumn)")


def evaluate_by_season_all_k(top_indices, target_dts, db_dts, k_list=TOP_K_LIST):
    """Computes all Top-k values in a single pass.

    Top-k lists are prefixes of each other, so per-neighbor MSE is computed once
    up to max(k) and each k averages a prefix. This is about 5x faster than one
    pass per k, which dominates the cost over all k, regions, and models.

    Neighbors missing from weather_map are skipped as in evaluate_by_season
    (single k): filtering happens *within* the first k, not by taking the first
    k after filtering.
    """
    kmax = max(k_list)
    acc = {k: {s: [] for s in SEASONS} for k in k_list}
    for i, t_dt in enumerate(target_dts):
        if t_dt not in weather_map: continue
        month = parse_month(t_dt)
        if month is None: continue
        season = month_to_season(month)
        target_w = np.asarray(weather_map[t_dt], dtype=np.float64)

        idxs = list(top_indices[i][:kmax])
        per_neighbor = np.full(len(idxs), np.nan)
        hit_pos = [j for j, di in enumerate(idxs) if db_dts[di] in weather_map]
        if hit_pos:
            vecs = np.asarray([weather_map[db_dts[idxs[j]]] for j in hit_pos], dtype=np.float64)
            per_neighbor[hit_pos] = ((vecs - target_w) ** 2).mean(axis=1)

        for k in k_list:
            vals = per_neighbor[:k]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                acc[k][season].append(vals.mean())
    return {k: {s: (np.mean(v) if v else np.nan, len(v)) for s, v in d.items()}
            for k, d in acc.items()}


def evaluate_by_season(top_indices, target_dts, db_dts, k):
    """Single-k wrapper kept for existing callers."""
    return evaluate_by_season_all_k(top_indices, target_dts, db_dts, [k])[k]


def run_seasonal_generalization(cfg=CONFIG):
    # Covers all of TOP_K_LIST so the table lines up with the other Top-1 to Top-100 results.
    season_rows = []
    for region_key, model_topidx in FINAL_REPORT_TOPIDX.items():
        for model_name, top_idx in model_topidx.items():
            per_k = evaluate_by_season_all_k(top_idx, query_eval_dts, db_eval_dts, TOP_K_LIST)
            for k in TOP_K_LIST:
                row = {"Region": region_key, "Model": model_name, "Top-k": k}
                for season, (mse, n) in per_k[k].items():
                    row[f"{season} MSE"] = mse
                    row[f"{season} N"] = n
                season_rows.append(row)

    season_df = pd.DataFrame(season_rows)
    season_k = min(TOP_K_LIST)
    print(f"Seasonal weather MSE (lower is better), Top-1 to Top-{max(TOP_K_LIST)}; rows below: Top-{season_k}")
    print(season_df[season_df["Top-k"] == season_k].to_string(index=False))

    # Max-min gap across seasons for the Proposed model (identifies the weakest
    # season). Judged at Top-1: larger k mixes neighbors and blurs seasonal contrast.
    proposed_rows = season_df[(season_df["Model"] == "Proposed") & (season_df["Top-k"] == season_k)]
    if len(proposed_rows) > 0:
        mse_cols = [c for c in season_df.columns if c.endswith("MSE")]
        for _, r in proposed_rows.iterrows():
            vals = {c.replace(" MSE", ""): r[c] for c in mse_cols if not pd.isna(r[c])}
            if vals:
                worst = max(vals, key=vals.get)
                best = min(vals, key=vals.get)
                print(f"\n[{r['Region']}] weakest season: {worst} (MSE={vals[worst]:.4f}) "
                      f"vs. best season: {best} (MSE={vals[best]:.4f}) "
                      f"| gap: {(vals[worst]-vals[best])/vals[best]:.1%}")

    out_path = os.path.join(run_dir(cfg), "seasonal_generalization.csv")
    season_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("\neval_mse_retrieval.run_evaluation() covers all four regions in REGIONS_TO_EVAL, "
          "so season_rows above includes every region.")
    return season_df


if __name__ == "__main__":
    run_seasonal_generalization()
