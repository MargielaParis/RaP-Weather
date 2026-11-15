"""LPIPS / PSNR / SSIM image-quality evaluation (Table 1) + statistical significance,
used to test whether small LPIPS gaps between models are significant."""
import os
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

from .config import CONFIG, device, embedding_save_path, run_dir
from .models import cleanup_gpu, load_trained_model, normalize_image
from .regions import REGIONS, region_tensor
from .eval_setup import SPLIT_IDX, TOP_K_LIST, bootstrap_ci, target_indices

loss_fn_alex = lpips.LPIPS(net='alex').to(device)

SCENARIOS = {
    "south_korea": {"indices": region_tensor("south_korea", device), "crop_box": REGIONS["south_korea"]["box"]},
    "total_patches": {"indices": region_tensor("total_patches", device), "crop_box": REGIONS["total_patches"]["box"]},
}


def calculate_metrics(target_np, retrieved_np):
    t_255 = (target_np * 255).astype(np.uint8) if target_np.max() <= 1.0 else target_np.astype(np.uint8)
    r_255 = (retrieved_np * 255).astype(np.uint8) if retrieved_np.max() <= 1.0 else retrieved_np.astype(np.uint8)
    psnr = psnr_func(t_255, r_255, data_range=255)
    min_side = min(t_255.shape[0], t_255.shape[1])
    win_size = min(7, min_side) if min_side % 2 != 0 else min(7, min_side - 1)
    ssim = ssim_func(t_255, r_255, data_range=255, channel_axis=2, win_size=win_size)
    t_tensor = (torch.tensor(t_255.transpose(2, 0, 1)).float() / 255.0 * 2 - 1).unsqueeze(0).to(device)
    r_tensor = (torch.tensor(r_255.transpose(2, 0, 1)).float() / 255.0 * 2 - 1).unsqueeze(0).to(device)
    with torch.no_grad():
        lpips_val = loss_fn_alex(t_tensor, r_tensor).item()
    return lpips_val, psnr, ssim


def run_table1_evaluation(cfg=CONFIG):
    table1_model = load_trained_model(cfg=cfg, device=device)
    with open(embedding_save_path(cfg), "rb") as f:
        prop_db_emb = pickle.load(f)['embeddings'].to(device)
    all_baselines = {}
    if os.path.exists(cfg["paths"]["baseline_pkl"]):
        with open(cfg["paths"]["baseline_pkl"], "rb") as f:
            all_baselines = pickle.load(f)

    final_summary, all_scenario_raw = [], {}   # all_scenario_raw: raw per-query lists for significance testing
    max_k = max(TOP_K_LIST)

    with h5py.File(cfg["paths"]["h5"], "r") as f_h5:
        images_dset = f_h5['images']
        for scenario_name, sc in SCENARIOS.items():
            print(f"\n\nEvaluating Scenario: {scenario_name.upper()}")
            curr_indices, (cx, cy, cw, ch) = sc['indices'], sc['crop_box']
            scenario_res = {"RaP-Weather": {m: {k: [] for k in TOP_K_LIST} for m in ["LPIPS", "PSNR", "SSIM"]}}
            if scenario_name in all_baselines:
                for vb in all_baselines[scenario_name].keys():
                    scenario_res[vb] = {m: {k: [] for k in TOP_K_LIST} for m in ["LPIPS", "PSNR", "SSIM"]}

            for target_idx in tqdm(target_indices, desc=f"Queries ({scenario_name})"):
                query_img_idx = target_idx - 1
                target_full = images_dset[query_img_idx]
                if target_full.shape[0] == 3: target_full = target_full.transpose(1, 2, 0)
                target_roi = target_full[cy:cy+ch, cx:cx+cw, :]

                img_tensor = normalize_image(torch.tensor(images_dset[query_img_idx], dtype=torch.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_emb = F.normalize(table1_model.forward_spatial(img_tensor), p=2, dim=2)
                    pred_region = pred_emb.index_select(dim=1, index=curr_indices)
                    db_region = F.normalize(prop_db_emb.index_select(dim=1, index=curr_indices), p=2, dim=2)
                    sims = torch.einsum('ipd, jpd -> ij', pred_region, db_region) / len(curr_indices)
                    _, top_idxs = torch.topk(sims, k=max_k, dim=1)
                    top_idxs = top_idxs[0].cpu().numpy()

                l_list, p_list, s_list = [], [], []
                for rank_idx in top_idxs:
                    retrieved_full = images_dset[rank_idx + cfg["seq_len"] - 1]
                    if retrieved_full.shape[0] == 3: retrieved_full = retrieved_full.transpose(1, 2, 0)
                    l, p, s = calculate_metrics(target_roi, retrieved_full[cy:cy+ch, cx:cx+cw, :])
                    l_list.append(l); p_list.append(p); s_list.append(s)
                for k in TOP_K_LIST:
                    scenario_res["RaP-Weather"]["LPIPS"][k].append(np.mean(l_list[:k]))
                    scenario_res["RaP-Weather"]["PSNR"][k].append(np.mean(p_list[:k]))
                    scenario_res["RaP-Weather"]["SSIM"][k].append(np.mean(s_list[:k]))

                if scenario_name in all_baselines:
                    for b_name, content in all_baselines[scenario_name].items():
                        feats = content.get('embeddings', content.get('feats')) if isinstance(content, dict) else content
                        if not torch.is_tensor(feats): feats = torch.tensor(feats)
                        q_vec = F.normalize(feats[query_img_idx].unsqueeze(0).to(device), p=2, dim=1)
                        db_vecs = F.normalize(feats[cfg["seq_len"]:SPLIT_IDX].to(device), p=2, dim=1)
                        sim = torch.mm(q_vec, db_vecs.T)
                        _, b_top_idxs = torch.topk(sim, k=max_k, dim=1)
                        b_top_idxs = b_top_idxs[0].cpu().numpy()

                        bl_list, bp_list, bs_list = [], [], []
                        for rank_idx in b_top_idxs:
                            r_full = images_dset[rank_idx + cfg["seq_len"]]
                            if r_full.shape[0] == 3: r_full = r_full.transpose(1, 2, 0)
                            l, p, s = calculate_metrics(target_roi, r_full[cy:cy+ch, cx:cx+cw, :])
                            bl_list.append(l); bp_list.append(p); bs_list.append(s)
                        for k in TOP_K_LIST:
                            scenario_res[b_name]["LPIPS"][k].append(np.mean(bl_list[:k]))
                            scenario_res[b_name]["PSNR"][k].append(np.mean(bp_list[:k]))
                            scenario_res[b_name]["SSIM"][k].append(np.mean(bs_list[:k]))

            all_scenario_raw[scenario_name] = scenario_res   # keep raw lists for significance testing
            for m_name, metrics in scenario_res.items():
                if len(metrics["LPIPS"][TOP_K_LIST[0]]) > 0:
                    row = {"Scenario": scenario_name, "Model": m_name}
                    for k in TOP_K_LIST:
                        row[f"LPIPS@{k}"] = np.mean(metrics["LPIPS"][k])
                        row[f"PSNR@{k}"] = np.mean(metrics["PSNR"][k])
                        row[f"SSIM@{k}"] = np.mean(metrics["SSIM"][k])
                    final_summary.append(row)

    del table1_model, prop_db_emb
    cleanup_gpu("table1_model", "prop_db_emb")
    df_final = pd.DataFrame(final_summary)
    model_order = ["RaP-Weather", "DINOv2", "CLIP", "ConvNeXt"]
    available = df_final['Model'].unique()
    adjusted_order = [m for m in model_order if m in available] + [m for m in available if m not in model_order]
    df_final['Model'] = pd.Categorical(df_final['Model'], categories=adjusted_order, ordered=True)
    df_final = df_final.sort_values(by=["Scenario", "Model"]).reset_index(drop=True)
    print("\nFinal Image Quality Evaluation Report (Top-K)")
    pd.set_option('display.max_columns', None)
    print(df_final)
    df_final.to_csv(os.path.join(run_dir(cfg), "table1_quality_comparison.csv"), index=False)
    print("\nSaved: table1_quality_comparison.csv")

    return df_final, all_scenario_raw


def run_table1_significance(all_scenario_raw, cfg=CONFIG):
    """### Statistical significance for Table 1 (LPIPS/PSNR/SSIM)."""
    metrics_to_test, baseline_model = ["LPIPS", "PSNR", "SSIM"], "RaP-Weather"
    table1_sig_rows = []
    for scenario_name, scenario_res in all_scenario_raw.items():
        if baseline_model not in scenario_res: continue
        for metric in metrics_to_test:
            base_by_k = scenario_res[baseline_model][metric]
            for model_name, metrics_dict in scenario_res.items():
                for k, vals in metrics_dict[metric].items():
                    vals = np.asarray(vals, dtype=np.float64)
                    if len(vals) == 0: continue
                    mean_v, lo, hi = bootstrap_ci(vals)
                    row = {"Scenario": scenario_name, "Metric": metric, "Model": model_name, "Top-k": k,
                           "Mean": mean_v, "95% CI Low": lo, "95% CI High": hi, "N": len(vals)}
                    if model_name != baseline_model and k in base_by_k:
                        base_vals = np.asarray(base_by_k[k], dtype=np.float64)
                        if len(base_vals) == len(vals) and len(vals) > 1:
                            try:
                                _, p = stats.wilcoxon(base_vals, vals)
                            except ValueError:
                                p = np.nan
                            row["Wilcoxon p (vs. RaP-Weather)"] = p
                            row["Significant (p<0.05)"] = bool(p < 0.05) if not np.isnan(p) else False
                    table1_sig_rows.append(row)

    table1_sig_df = pd.DataFrame(table1_sig_rows)
    print("Table 1 statistical significance (bootstrap 95% CI + paired Wilcoxon vs. RaP-Weather)")
    print(table1_sig_df.to_string(index=False))
    table1_sig_df.to_csv(os.path.join(run_dir(cfg), "table1_significance.csv"), index=False)
    print("\nSaved: table1_significance.csv")
    return table1_sig_df


def main():
    df_final, all_scenario_raw = run_table1_evaluation()
    run_table1_significance(all_scenario_raw)


if __name__ == "__main__":
    main()
