"""Meteorological-MSE retrieval evaluation + statistical significance, across all regions.

RaP-Weather ("Proposed") is scored with mean per-patch cosine over the region's
patches (paper Eq. 9, calculate_patch_mse_metrics); precomputed baselines use
global cosine between single vectors (calculate_mse_metrics).

FINAL_REPORT / FINAL_REPORT_RAW / FINAL_REPORT_TOPIDX are populated in place by
run_evaluation() and are then reused (merged into) by later modules such as
the baselines/ Group 1/2 comparisons.
"""
import os
import pickle

import h5py
import pandas as pd
import torch
from tqdm import tqdm

from .config import CONFIG, device, run_dir
from .models import cleanup_gpu, load_trained_model, normalize_image
from .regions import REGIONS_TO_EVAL, region_tensor
from .eval_setup import (
    TOP_K_LIST, SPLIT_IDX, db_indices, target_indices, db_eval_dts, query_eval_dts,
    calculate_mse_metrics, calculate_patch_mse_metrics, paired_significance_report,
)

# Populated by run_evaluation(); mutated in place by later modules (e.g. Group1, Group2)
# via `FINAL_REPORT[region][model] = ...` item assignment.
FINAL_REPORT, FINAL_REPORT_RAW, FINAL_REPORT_TOPIDX = {}, {}, {}


def run_evaluation(cfg=CONFIG):
    for region_key in REGIONS_TO_EVAL:
        print(f"\n\n{'='*40}\nProcessing Region: {region_key.upper()}\n{'='*40}")
        region_results, region_results_raw, region_top_idx = {}, {}, {}

        # (A) Proposed model
        print("\nEvaluating Proposed Model...")
        eval_model = load_trained_model(cfg=cfg, device=device)
        curr_indices = region_tensor(region_key, device)

        prop_db_patches, prop_query_patches = [], []
        with h5py.File(cfg["paths"]["h5"], "r") as f:
            dset = f['images']
            with torch.no_grad():
                for start_b in tqdm(range(cfg["seq_len"], SPLIT_IDX, cfg["batch_size"]), desc="Extracting Proposed DB (t)"):
                    end_b = min(start_b + cfg["batch_size"], SPLIT_IDX)
                    img_tensor = normalize_image(torch.tensor(dset[start_b:end_b], dtype=torch.float32)).to(device)
                    embs = eval_model.forward_spatial(img_tensor)
                    prop_db_patches.append(embs.index_select(1, curr_indices).cpu())
            prop_db_patches = torch.cat(prop_db_patches, dim=0).to(device)

            with torch.no_grad():
                for idx in tqdm(target_indices, desc="Extracting Proposed Queries (t)"):
                    img_tensor = normalize_image(torch.tensor(dset[idx - 1], dtype=torch.float32).unsqueeze(0)).to(device)
                    emb = eval_model.forward_spatial(img_tensor)
                    prop_query_patches.append(emb.index_select(1, curr_indices).cpu())
            prop_query_patches = torch.cat(prop_query_patches, dim=0).to(device)

        res, res_raw, res_topidx = calculate_patch_mse_metrics(prop_query_patches, prop_db_patches, query_eval_dts, db_eval_dts)
        region_results["Proposed"] = res
        region_results_raw["Proposed"] = res_raw
        region_top_idx["Proposed"] = res_topidx
        del eval_model, prop_db_patches, prop_query_patches
        cleanup_gpu("eval_model", "prop_db_patches", "prop_query_patches")

        # (B) Baselines
        print("\nEvaluating Baselines...")
        if os.path.exists(cfg["paths"]["baseline_pkl"]):
            with open(cfg["paths"]["baseline_pkl"], 'rb') as f:
                all_baselines = pickle.load(f)
            if region_key in all_baselines:
                for model_name, content in all_baselines[region_key].items():
                    print(f"   -> Model: {model_name}")
                    feats = content.get('embeddings', content.get('feats')) if isinstance(content, dict) else content
                    if feats is None: continue
                    if not torch.is_tensor(feats): feats = torch.tensor(feats)
                    base_db_vecs = feats[db_indices].to(device)
                    base_query_vecs = feats[target_indices - 1].to(device)
                    res, res_raw, res_topidx = calculate_mse_metrics(base_query_vecs, base_db_vecs, query_eval_dts, db_eval_dts)
                    region_results[model_name] = res
                    region_results_raw[model_name] = res_raw
                    region_top_idx[model_name] = res_topidx
                    del base_db_vecs, base_query_vecs
                    cleanup_gpu("base_db_vecs", "base_query_vecs")

        FINAL_REPORT[region_key] = region_results
        FINAL_REPORT_RAW[region_key] = region_results_raw
        FINAL_REPORT_TOPIDX[region_key] = region_top_idx

    for region_key, results in FINAL_REPORT.items():
        print(f"\n\nFinal Results: {region_key.upper()}")
        df = pd.DataFrame(results).T
        df.columns = [f"Top-{k}" for k in TOP_K_LIST]
        print(df)
        df.to_csv(os.path.join(run_dir(cfg), f"mse_comparison_{region_key}_D{cfg['patch_emb_size']}.csv"))


def run_significance(cfg=CONFIG):
    """### Statistical significance across all regions evaluated in run_evaluation()."""
    for region_key in FINAL_REPORT_RAW.keys():
        print(f"\n\n{'='*60}\nStatistical Significance vs. Proposed - {region_key.upper()}\n{'='*60}")
        sig_df = paired_significance_report(FINAL_REPORT_RAW[region_key], baseline_name="Proposed")
        print(sig_df.to_string(index=False))
        sig_out = os.path.join(run_dir(cfg), f"significance_{region_key}.csv")
        sig_df.to_csv(sig_out, index=False)
        print(f"Saved: {sig_out}")


def save_top_indices(cfg=CONFIG):
    """Save FINAL_REPORT_TOPIDX as one npz per region.

    With the rankings on disk, re-aggregating by season or by k takes seconds
    on CPU instead of re-running the full (GPU-heavy) evaluation stage.
    """
    import numpy as np
    saved = []
    for region_key, model_topidx in FINAL_REPORT_TOPIDX.items():
        if not model_topidx:
            continue
        names = list(model_topidx.keys())
        out = os.path.join(run_dir(cfg), f"top_indices_{region_key}.npz")
        np.savez_compressed(
            out,
            model_names=np.array(names),   # unicode array; loads without allow_pickle
            **{f"m{i}": np.asarray(model_topidx[n]) for i, n in enumerate(names)},
        )
        saved.append((region_key, len(names), out))
    for region_key, n, out in saved:
        print(f"Saved top-k rankings ({n} models): {out}")
    return [s[2] for s in saved]


def main():
    run_evaluation()
    save_top_indices()
    run_significance()


if __name__ == "__main__":
    main()
