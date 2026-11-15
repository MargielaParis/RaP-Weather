"""Paper Group 1: off-the-shelf baselines (CLIP / ConvNeXt2 / DINOv2), no training.

The region is cropped from each image and encoded on the fly with a frozen
backbone (traditional CBIR, one vector per image, no patch grid). DB and query
embeddings are scored with eval_setup.calculate_mse_metrics, i.e. global cosine
similarity between single vectors, and the Top-k weather MSE is reported.
Inputs are [0,1] tensors without per-backbone mean/std normalization.
"""
import os

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..config import CONFIG, device, run_dir
from ..models import normalize_image
from ..regions import REGIONS, REGIONS_TO_EVAL
from ..eval_setup import TOP_K_LIST, calculate_mse_metrics, db_eval_dts, db_indices, query_eval_dts, target_indices
from ..eval_mse_retrieval import FINAL_REPORT, FINAL_REPORT_RAW, FINAL_REPORT_TOPIDX
from .backbone_registry import BACKBONE_SPECS, crop_and_resize, get_frozen_backbone


# =========================================================
# Group 1 evaluation: off-the-shelf backbone, region crop encoded on the fly.
# h5py fancy indexing requires increasing indices: sort, read, then restore order.
# =========================================================
def extract_group1_embeddings(backbone_name, indices, region_key="south_korea", cfg=CONFIG, desc=""):
    model, input_size, feat_dim = get_frozen_backbone(backbone_name)
    box = REGIONS[region_key]["box"]

    indices = np.asarray(indices)
    order = np.argsort(indices)
    sorted_indices = indices[order]

    embs_sorted = []
    with h5py.File(cfg["paths"]["h5"], "r") as f, torch.no_grad():
        dset = f['images']
        for start in tqdm(range(0, len(sorted_indices), cfg["batch_size"]), desc=f"[Group1-{backbone_name}/{region_key}] {desc}"):
            batch_idx = sorted_indices[start:start + cfg["batch_size"]]
            imgs = normalize_image(torch.tensor(dset[list(batch_idx)], dtype=torch.float32)).to(device)
            crops = crop_and_resize(imgs, box, input_size)
            feat = model(crops)
            embs_sorted.append(feat.cpu())
    embs_sorted = torch.cat(embs_sorted, dim=0)

    inverse_order = np.argsort(order)
    return embs_sorted[inverse_order]


def evaluate_group1(backbone_name, region_key="south_korea", cfg=CONFIG):
    db_embs = extract_group1_embeddings(backbone_name, db_indices, region_key, cfg, desc="DB")
    query_embs = extract_group1_embeddings(backbone_name, (target_indices - 1), region_key, cfg, desc="Query")
    mean_res, raw_res, top_idx = calculate_mse_metrics(
        query_embs.to(device), db_embs.to(device), query_eval_dts, db_eval_dts
    )
    model_key = f"{backbone_name} (Group1, off-the-shelf)"
    if region_key in FINAL_REPORT_RAW:
        FINAL_REPORT[region_key][model_key] = mean_res
        FINAL_REPORT_RAW[region_key][model_key] = raw_res
        FINAL_REPORT_TOPIDX[region_key][model_key] = top_idx
    return mean_res, raw_res, top_idx


def main(cfg=CONFIG):
    # Evaluate every region x backbone pair.
    group1_results = {}
    for region_key in REGIONS_TO_EVAL:
        group1_results[region_key] = {}
        for backbone_name in BACKBONE_SPECS.keys():
            print(f"\n{'='*50}\nGroup 1 - {backbone_name} / {region_key}\n{'='*50}")
            mean_res, raw_res, top_idx = evaluate_group1(backbone_name, region_key=region_key, cfg=cfg)
            group1_results[region_key][backbone_name] = mean_res
            for k in TOP_K_LIST:
                print(f"   Top-{k}: MSE = {mean_res[k]:.4f}")

    for region_key, res in group1_results.items():
        group1_df = pd.DataFrame(res).T
        group1_df.columns = [f"Top-{k}" for k in TOP_K_LIST]
        print(f"\n[{region_key}] Group 1 summary:")
        print(group1_df)
        group1_df.to_csv(os.path.join(run_dir(cfg), f"group1_offtheshelf_{region_key}.csv"))
        print(f"Saved: group1_offtheshelf_{region_key}.csv")

    print("\nResults merged into FINAL_REPORT_RAW / FINAL_REPORT_TOPIDX (per region) for the unified comparison.")
    return group1_results


if __name__ == "__main__":
    main()
