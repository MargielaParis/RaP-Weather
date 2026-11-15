"""Confirmatory statistics from saved per-query results (Tables 2 to 6).

Reads the per-query retrieval results written by the evaluation scripts and the
raw weather vectors (export_weather_vectors.py), then computes:
  * per-query bootstrap 95% CIs (2,000 resamples, seed 42),
  * paired Wilcoxon signed-rank tests against RaP-Weather,
  * Holm correction within each comparison family.
The 200 model-selection queries (seed 42: drop a legacy 500-query report set;
seed 123: draw 200 from the rest) are removed, leaving N = 4,943.

The visual-metric family includes Group 1 (off-the-shelf) and the paper's
Group 2. In code, Group 2 is "G3" for regional crops and "Group2" for the full
image; see the README for the naming map.

No model is retrained or re-ranked. Outputs go to the stats dir (results/).
"""
from __future__ import annotations

import json
import pickle
from collections import OrderedDict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


from _paths import DATA_DIR as DATA, FIGURES_DIR, RUN, STATS_DIR as HERE, WEATHER_NPZ

HERE.mkdir(parents=True, exist_ok=True)

TOP_K = (1, 3, 5, 10, 50, 100)
REGIONS = ("south_korea", "south_korea_wide", "low_left", "total_patches")
BACKBONES = ("CLIP", "ConvNeXt2", "DINOv2")
G1 = "{} (Group1, off-the-shelf)"
G3_RAW = "{} (G3: whole-trained, crop@query)"
MAIN_G2 = "{} (Group2, whole-trained + crop)"
MAIN_G2_TOTAL = "{} (Group2, whole-image fine-tuned)"
PER_REGION = "{} (per-region)"
PIXEL = "Pixel-NN"

N_TOTAL = 25_744
SEQ_LEN = 6
SPLIT = int(N_TOTAL * 0.8)
CANDIDATES = np.arange(SPLIT + SEQ_LEN, N_TOTAL)
N_BOOT = 2_000
BOOT_SEED = 42


def reconstruct_selection_queries() -> np.ndarray:
    legacy_report = np.random.RandomState(42).choice(CANDIDATES, 500, replace=False)
    pool = np.setdiff1d(CANDIDATES, legacy_report)
    selected = np.random.RandomState(123).choice(pool, 200, replace=False)
    if len(selected) != 200 or not np.all(np.isin(selected, CANDIDATES)):
        raise RuntimeError("failed to reconstruct the 200 model-selection queries")
    return np.sort(selected)


def clean_as_reported(vectors: np.ndarray) -> np.ndarray:
    med = np.median(vectors, axis=0)
    mad = np.median(np.abs(vectors - med), axis=0)
    scale = np.where(mad > 0, 1.4826 * mad, 1.0)
    mask = (np.abs(vectors - med) / scale > 20.0) | (vectors <= -99.0)
    cleaned = vectors.astype(np.float64, copy=True)
    cleaned[mask] = np.nan
    col_median = np.nanmedian(cleaned, axis=0)
    idx = np.where(mask)
    cleaned[idx] = np.take(col_median, idx[1])
    if int(mask.sum()) != 488 or int(mask.any(axis=1).sum()) != 441:
        raise RuntimeError(f"sentinel audit changed: values={mask.sum()}, rows={mask.any(axis=1).sum()}")
    return cleaned


def load_weather_arrays():
    raw = np.load(WEATHER_NPZ)
    weather_dates = raw["datetimes"].astype(str)
    vectors = clean_as_reported(raw["vectors"])
    with h5py.File(DATA / "coms_25744_224.hdf5", "r") as handle:
        image_dates = handle["datetimes"][:].astype(str)
    if not np.array_equal(weather_dates, image_dates):
        raise RuntimeError("weather and image timestamps are not aligned")
    with (DATA / "weather_scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    if int(scaler.n_samples_seen_) != N_TOTAL or int(scaler.n_features_in_) != 112:
        raise RuntimeError("unexpected cached weather scaler")
    scaled = (vectors - scaler.mean_) / scaler.scale_
    db_indices = np.arange(SEQ_LEN, SPLIT)
    query_image_indices = CANDIDATES - 1
    return scaled[query_image_indices], scaled[db_indices], image_dates[query_image_indices], image_dates[db_indices]


def perquery_from_rankings(rank, weather_q, weather_db):
    mse = ((weather_q[:, None, :] - weather_db[rank]) ** 2).mean(axis=2)
    cumulative = np.cumsum(mse, axis=1) / np.arange(1, mse.shape[1] + 1)
    return {k: cumulative[:, k - 1] for k in TOP_K}


def load_ranked_models(region, weather_q, weather_db):
    archive = np.load(RUN / f"top_indices_{region}.npz", allow_pickle=True)
    names = [str(name) for name in archive["model_names"]]
    result = OrderedDict()
    for name in ["Proposed", *(G1.format(bb) for bb in BACKBONES)]:
        result[name] = perquery_from_rankings(archive[f"m{names.index(name)}"], weather_q, weather_db)
    return result


def load_main_weather(region, weather_q, weather_db):
    result = load_ranked_models(region, weather_q, weather_db)
    if region == "total_patches":
        archive = np.load(RUN / "group2_fulldim_perquery_total_patches.npz")
        for bb in BACKBONES:
            result[MAIN_G2_TOTAL.format(bb)] = {k: archive[f"{bb}||Top-{k}"].astype(np.float64) for k in TOP_K}
    else:
        archive = np.load(RUN / f"group3_fulldim_perquery_raw_{region}.npz")
        for bb in BACKBONES:
            result[MAIN_G2.format(bb)] = {k: archive[f"{G3_RAW.format(bb)}||Top-{k}"].astype(np.float64) for k in TOP_K}
    if region == "south_korea":
        archive = np.load(RUN / "pixel_nn_perquery_south_korea.npz")
        result[PIXEL] = {k: archive[f"Pixel-NN||Top-{k}"].astype(np.float64) for k in TOP_K}
    return result


def load_per_region_weather(region, proposed):
    archive = np.load(RUN / f"group2_fulldim_perquery_{region}.npz")
    result = OrderedDict([("Proposed", proposed)])
    for bb in BACKBONES:
        result[PER_REGION.format(bb)] = {k: archive[f"{bb}||Top-{k}"].astype(np.float64) for k in TOP_K}
    return result


# ---------------- per-query bootstrap + per-query paired Wilcoxon + Holm ----------------

def query_ci(values, seed: int = BOOT_SEED, n_boot: int = N_BOOT):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, n, size=(n_boot, n))
    boot = values[draws].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def paired_wilcoxon(base, other) -> float:
    try:
        return float(wilcoxon(np.asarray(base, float), np.asarray(other, float)).pvalue)
    except ValueError:
        return 1.0


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, (total - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def stats_for_family(datasets, keep) -> pd.DataFrame:
    rows, test_rows, raw_ps = [], [], []
    for scope, models in datasets.items():
        base = models["Proposed"]
        for model, per_k in models.items():
            for k in TOP_K:
                values = np.asarray(per_k[k])[keep]
                mean, low, high = query_ci(values)
                row = {"Scope": scope, "Model": model, "Top-k": k, "Mean": mean,
                       "95% CI Low": low, "95% CI High": high, "N queries": len(values)}
                if model != "Proposed":
                    base_values = np.asarray(base[k])[keep]
                    p = paired_wilcoxon(base_values, values)
                    row["Wilcoxon p"] = p
                    row["Baseline mean lower"] = bool(mean < base_values.mean())
                    test_rows.append(len(rows))
                    raw_ps.append(p)
                rows.append(row)
    frame = pd.DataFrame(rows)
    adjusted = holm_adjust(np.asarray(raw_ps))
    for row_idx, p_adj in zip(test_rows, adjusted):
        frame.loc[row_idx, "Holm p"] = p_adj
        frame.loc[row_idx, "Holm significant"] = bool(p_adj < 0.05)
    return frame


def load_visual(query_type: str):
    """Group 1 (off-the-shelf) + paper Group 2 (code Group3 for Region, Group2 for Full Image)."""
    suffix = "south_korea" if query_type == "Region" else "total_patches"
    archive = np.load(RUN / f"table1_group13_perquery_{suffix}_fulldim.npz")
    models = OrderedDict()
    names = ["RaP-Weather"]
    names.extend(f"{bb} (Group1, off-the-shelf)" for bb in BACKBONES)
    if query_type == "Region":
        names.extend(f"{bb} (Group3, whole-trained + crop)" for bb in BACKBONES)
    else:
        names.extend(f"{bb} (Group2, predictive fine-tuned)" for bb in BACKBONES)
    for name in names:
        models[name] = {metric: {k: archive[f"{name}||{metric}||{k}"].astype(np.float64) for k in TOP_K}
                        for metric in ("LPIPS", "PSNR", "SSIM")}
    return models


def visual_stats(keep) -> pd.DataFrame:
    rows, test_rows, raw_ps = [], [], []
    for query_type in ("Full Image", "Region"):
        models = load_visual(query_type)
        base = models["RaP-Weather"]
        for metric in ("LPIPS", "PSNR", "SSIM"):
            for model, per_metric in models.items():
                for k in TOP_K:
                    values = per_metric[metric][k][keep]
                    mean, low, high = query_ci(values)
                    row = {"Query Type": query_type, "Metric": metric, "Model": model, "Top-k": k,
                           "Mean": mean, "95% CI Low": low, "95% CI High": high, "N queries": len(values)}
                    if model != "RaP-Weather":
                        bv = base[metric][k][keep]
                        p = paired_wilcoxon(bv, values)
                        row["Wilcoxon p"] = p
                        row["Baseline better"] = bool(mean < bv.mean()) if metric == "LPIPS" else bool(mean > bv.mean())
                        test_rows.append(len(rows))
                        raw_ps.append(p)
                    rows.append(row)
    frame = pd.DataFrame(rows)
    adjusted = holm_adjust(np.asarray(raw_ps))
    for row_idx, p_adj in zip(test_rows, adjusted):
        frame.loc[row_idx, "Holm p"] = p_adj
        frame.loc[row_idx, "Holm significant"] = bool(p_adj < 0.05)
    return frame


from plot_fig6 import plot_main_top1  # figure sized for print; see plot_fig6.py


def main() -> None:
    selection = reconstruct_selection_queries()
    keep = ~np.isin(CANDIDATES, selection)
    if int(keep.sum()) != 4_943:
        raise RuntimeError(f"unexpected confirmatory query count: {keep.sum()}")

    weather_q, weather_db, query_dates, _ = load_weather_arrays()
    kept_dates = query_dates[keep]

    main_data, per_region_data = OrderedDict(), OrderedDict()
    for region in REGIONS:
        models = load_main_weather(region, weather_q, weather_db)
        main_data[region] = models
        per_region_data[region] = load_per_region_weather(region, models["Proposed"])

    main_frame = stats_for_family(main_data, keep)
    per_region_frame = stats_for_family(per_region_data, keep)
    visual_frame = visual_stats(keep)

    main_frame.to_csv(HERE / "primary_main_stats.csv", index=False)
    per_region_frame.to_csv(HERE / "per_region_reference_stats.csv", index=False)
    visual_frame.to_csv(HERE / "visual_metric_stats.csv", index=False)
    pd.DataFrame({"candidate_index": selection,
                  "query_datetime": query_dates[np.isin(CANDIDATES, selection)]}
                 ).to_csv(HERE / "model_selection_queries.csv", index=False)

    comparisons = main_frame[main_frame["Model"] != "Proposed"].copy()
    cells = []
    for (scope, k), group in comparisons.groupby(["Scope", "Top-k"], sort=False):
        proposed_mean = float(main_frame[(main_frame["Scope"] == scope) & (main_frame["Top-k"] == k)
                                         & (main_frame["Model"] == "Proposed")]["Mean"].iloc[0])
        all_better = bool(((group["Holm p"] < 0.05) & (group["Mean"] > proposed_mean)).all())
        significant_loss = bool(((group["Holm p"] < 0.05) & (group["Mean"] < proposed_mean)).any())
        lowest_mean = bool((group["Mean"] > proposed_mean).all())
        losers = group[(group["Holm p"] < 0.05) & (group["Mean"] < proposed_mean)]["Model"].tolist()
        cells.append({"Scope": scope, "Top-k": k, "RaP lowest mean": lowest_mean,
                      "RaP beats all": all_better, "RaP significant loss": significant_loss,
                      "Significant winners over RaP": ";".join(losers)})
    cells_frame = pd.DataFrame(cells)
    cells_frame.to_csv(HERE / "primary_cell_claims.csv", index=False)

    # per-region reference cells (Appendix C claim sentences)
    pr_comp = per_region_frame[per_region_frame["Model"] != "Proposed"].copy()
    pr_cells = []
    for (scope, k), group in pr_comp.groupby(["Scope", "Top-k"], sort=False):
        pm = float(per_region_frame[(per_region_frame["Scope"] == scope) & (per_region_frame["Top-k"] == k)
                                    & (per_region_frame["Model"] == "Proposed")]["Mean"].iloc[0])
        winners = group[(group["Holm p"] < 0.05) & (group["Mean"] < pm)]["Model"].tolist()
        lower = group[group["Mean"] < pm]["Model"].tolist()
        pr_cells.append({"Scope": scope, "Top-k": k, "Per-region lower mean": ";".join(lower),
                         "Per-region significantly better": ";".join(winners),
                         "RaP beats all per-region": bool(((group["Holm p"] < 0.05) & (group["Mean"] > pm)).all())})
    pd.DataFrame(pr_cells).to_csv(HERE / "per_region_cell_claims.csv", index=False)

    summary = {
        "protocol": "per-query bootstrap (2000, seed 42) + per-query paired Wilcoxon + Holm per family",
        "future_pool_queries": int(len(CANDIDATES)),
        "model_selection_queries_removed": int((~keep).sum()),
        "confirmatory_queries": int(keep.sum()),
        "main_family_tests": int(len(comparisons)),
        "main_family_holm_significant": int((comparisons["Holm p"] < 0.05).sum()),
        "cells_rap_lowest_mean": int(cells_frame["RaP lowest mean"].sum()),
        "cells_rap_beats_all": int(cells_frame["RaP beats all"].sum()),
        "cells_rap_significant_loss": int(cells_frame["RaP significant loss"].sum()),
        "per_region_family_tests": int(len(pr_comp)),
        "per_region_cells_with_significant_winner": int(sum(bool(c["Per-region significantly better"]) for c in pr_cells)),
        "visual_family_tests": int((visual_frame["Model"] != "RaP-Weather").sum()),
        "selection_first_datetime": str(query_dates[np.isin(CANDIDATES, selection)][0]),
        "selection_last_datetime": str(query_dates[np.isin(CANDIDATES, selection)][-1]),
        "confirmatory_first_datetime": str(kept_dates[0]),
        "confirmatory_last_datetime": str(kept_dates[-1]),
    }
    (HERE / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_main_top1(main_frame, FIGURES_DIR / "fig_unified_g123.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
