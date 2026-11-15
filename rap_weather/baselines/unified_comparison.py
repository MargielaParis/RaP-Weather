"""Unified quantitative comparison: RaP-Weather ("Proposed") vs. paper Group 1
(code Group1, off-the-shelf) vs. the per-region encoders (code Group2,
Appendix B), with weather MSE and paired significance tests.

Reuses paired_significance_report() from eval_setup. The Group 1/2 evaluation
functions already merge their results into FINAL_REPORT_RAW[region] and
FINAL_REPORT_TOPIDX[region], so the full comparison needs no extra code.

## Unified per-index comparison: Proposed + 3 Group 1 + 3 Group 2 models, any query index

A single model_name string selects the right encoder and similarity:
RaP-Weather uses its patch grid and mean per-patch cosine (paper Eq. 9);
Group 1/2 use a single vector and global cosine. DB embeddings are computed
once per model and cached.
"""
import os

import h5py
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..config import CONFIG, device, run_dir
from ..models import load_trained_model, normalize_image
from ..regions import REGIONS, REGIONS_TO_EVAL, region_tensor
from ..eval_setup import SPLIT_IDX, db_eval_dts, db_indices, total_samples, weather_map
from ..eval_mse_retrieval import FINAL_REPORT, FINAL_REPORT_RAW, FINAL_REPORT_TOPIDX
from .backbone_registry import BACKBONE_SPECS, crop_and_resize, get_frozen_backbone
from .group2_finetuned_baseline import WholeRegionPredictor, group2_checkpoint_path


def run_unified_mse_and_significance(cfg=CONFIG):
    from ..eval_setup import paired_significance_report, TOP_K_LIST

    # All regions: weather MSE + significance only. Precision/Recall/MAP are not
    # reported because the dataset has no ground-truth relevance labels.
    for region_key in REGIONS_TO_EVAL:
        if region_key not in FINAL_REPORT_RAW or not FINAL_REPORT_RAW[region_key]:
            print(f"[{region_key}] no results; skipping")
            continue

        print(f"\n[{region_key}] registered models: {list(FINAL_REPORT_RAW[region_key].keys())}\n")

        # 1) Weather MSE summary table
        mse_summary_df = pd.DataFrame(FINAL_REPORT[region_key]).T
        mse_summary_df.columns = [f"Top-{k}" for k in TOP_K_LIST]
        print("="*90 + f"\n[{region_key}] [1] Weather MSE comparison (Proposed + Group1 + Group2)\n" + "="*90)
        print(mse_summary_df)
        mse_summary_df.to_csv(os.path.join(run_dir(cfg), f"unified_mse_comparison_{region_key}.csv"))

        # 2) Statistical significance vs. Proposed
        print("\n" + "="*90 + f"\n[{region_key}] [2] Statistical significance vs. Proposed (bootstrap CI + Wilcoxon)\n" + "="*90)
        unified_sig_df = paired_significance_report(FINAL_REPORT_RAW[region_key], baseline_name="Proposed")
        print(unified_sig_df.to_string(index=False))
        unified_sig_df.to_csv(os.path.join(run_dir(cfg), f"unified_significance_{region_key}.csv"), index=False)

        print(f"\nSaved: unified_mse_comparison_{region_key}.csv, unified_significance_{region_key}.csv")

        # 3) Bar chart of Top-1 MSE, colored by group
        fig, ax = plt.subplots(figsize=(12, 6))
        k1 = f"Top-{TOP_K_LIST[0]}"
        colors = ["#1f77b4" if m == "Proposed" else ("#2ca02c" if "Group1" in m else ("#d62728" if "Group2" in m else "#888888"))
                  for m in mse_summary_df.index]
        ax.bar(mse_summary_df.index, mse_summary_df[k1], color=colors, edgecolor='black')
        ax.set_ylabel(f"{k1} MSE (Lower is Better)")
        ax.set_title(f"Region: {region_key} - Proposed (blue) vs Group1 (green, off-the-shelf) vs Group2 (red, fine-tuned)")
        plt.xticks(rotation=30, ha='right'); ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir(cfg), f"unified_comparison_bar_{region_key}.png"), dpi=300)
        plt.show()



# =========================================================
# Unified per-index comparison: dispatch Proposed (patch) vs. Group1/Group2 (vector)
# =========================================================
_UNIFIED_DB_CACHE = {}


def _get_model_encoder(model_name, region_key="south_korea", cfg=CONFIG):
    """model_name -> (encode_fn(img_batch) -> embedding, kind), kind in {'patch', 'vector'}."""
    box = REGIONS[region_key]["box"]

    if model_name == "Proposed":
        model = load_trained_model(cfg=cfg, device=device)
        curr_indices = region_tensor(region_key, device)
        return (lambda imgs: model.forward_spatial(imgs).index_select(1, curr_indices)), "patch"

    for backbone_name in BACKBONE_SPECS:
        if model_name == f"{backbone_name} (Group1, off-the-shelf)":
            frozen_model, input_size, _ = get_frozen_backbone(backbone_name)
            return (lambda imgs, m=frozen_model, s=input_size: m(crop_and_resize(imgs, box, s))), "vector"

        if model_name == f"{backbone_name} (Group2, predictive fine-tuned)":
            ckpt = group2_checkpoint_path(backbone_name, region_key, cfg)
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"{ckpt} not found; run train_group2_baseline('{backbone_name}') first.")
            g2 = WholeRegionPredictor(backbone_name, embed_dim=cfg["patch_emb_size"], seq_len=cfg["seq_len"]).to(device)
            g2.load_state_dict(torch.load(ckpt, map_location=device)); g2.eval()
            return (lambda imgs, m=g2: m.forward_spatial(imgs, box)), "vector"

    raise ValueError(f"Unknown model_name: {model_name}")


def _get_or_build_db_embeddings(model_name, region_key="south_korea", cfg=CONFIG):
    key = (model_name, region_key, cfg["patch_emb_size"])
    if key in _UNIFIED_DB_CACHE:
        return _UNIFIED_DB_CACHE[key]
    encode_fn, kind = _get_model_encoder(model_name, region_key, cfg)
    embs = []
    with h5py.File(cfg["paths"]["h5"], "r") as f, torch.no_grad():
        dset = f['images']
        for start_b in range(cfg["seq_len"], SPLIT_IDX, cfg["batch_size"]):
            end_b = min(start_b + cfg["batch_size"], SPLIT_IDX)
            imgs = normalize_image(torch.tensor(dset[start_b:end_b], dtype=torch.float32)).to(device)
            embs.append(encode_fn(imgs).cpu())
    embs = torch.cat(embs, dim=0).to(device)
    _UNIFIED_DB_CACHE[key] = (encode_fn, kind, embs)
    print(f"[{model_name}] cached DB embeddings ({len(embs)} items, kind={kind})")
    return _UNIFIED_DB_CACHE[key]


def inspect_raw_index_all_models(query_img_idx, region_key="south_korea", top_k_show=5,
                                  model_names=None, cfg=CONFIG):
    """query_img_idx: raw HDF5 index of the observed future time step.
       Must lie in the query pool [SPLIT_IDX + seq_len, total_samples): the
       intended use is retrieving similar past cases (DB) for a future query.
       As in eval_mse_retrieval, the image at idx-1 is encoded.
       With model_names=None, every model registered in
       FINAL_REPORT_TOPIDX[region_key] (Proposed + 3 Group1 + 3 Group2, only
       those already evaluated) is compared."""
    valid_lo = SPLIT_IDX + cfg["seq_len"]
    valid_hi = total_samples
    assert valid_lo <= query_img_idx < valid_hi, (
        f"query_img_idx must be in [{valid_lo}, {valid_hi}) (query pool = final 20%). "
        f"Pass the future query time step, not a past DB index."
    )
    model_names = model_names or list(FINAL_REPORT_TOPIDX.get(region_key, {"Proposed": None}).keys())
    box = REGIONS[region_key]["box"]
    bx, by, bw, bh = box

    with h5py.File(cfg["paths"]["h5"], "r") as f:
        dset = f['images']
        query_img = dset[query_img_idx - 1]   # same convention as eval_mse_retrieval: image at t is idx-1
        if query_img.shape[0] == 3: query_img = query_img.transpose(1, 2, 0)
        dt_raw = f['datetimes'][query_img_idx - 1]
        query_dt = dt_raw.decode() if isinstance(dt_raw, bytes) else str(dt_raw)
    print(f"Raw index: {query_img_idx} (encode idx-1={query_img_idx-1})  |  datetime: {query_dt}  |  models: {model_names}")

    fig, axes = plt.subplots(len(model_names), top_k_show + 1, figsize=(3.6 * (top_k_show + 1), 3.6 * len(model_names)))
    if len(model_names) == 1:
        axes = axes[None, :]

    for row, model_name in enumerate(model_names):
        try:
            encode_fn, kind, db_embs = _get_or_build_db_embeddings(model_name, region_key, cfg)
        except FileNotFoundError as e:
            print(f"[{model_name}] skipped: {e}")
            for ax in axes[row]: ax.axis('off')
            continue

        with h5py.File(cfg["paths"]["h5"], "r") as f, torch.no_grad():
            dset = f['images']
            img_t = normalize_image(torch.tensor(dset[query_img_idx - 1], dtype=torch.float32).unsqueeze(0)).to(device)
            q_emb = encode_fn(img_t)

        if kind == "patch":
            sims = (F.normalize(db_embs, p=2, dim=2) * F.normalize(q_emb, p=2, dim=2)).sum(dim=2).mean(dim=1)
        else:
            sims = (F.normalize(db_embs, p=2, dim=1) * F.normalize(q_emb, p=2, dim=1)).sum(dim=1)

        top_scores, top_ranks = torch.topk(sims, k=top_k_show)
        top_ranks = top_ranks.cpu().numpy()

        axes[row, 0].imshow(query_img)
        axes[row, 0].add_patch(patches.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='red', facecolor='none'))
        axes[row, 0].set_title(f"Query (t)\n{query_dt}\n[{model_name}]", fontsize=8, fontweight='bold')
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
                ax.set_title(f"Top-{col+1}\n{found_dt}\nSim:{top_scores[col]:.3f}\nMSE:{mse_str}",
                              fontsize=7.5, color=title_col, fontweight='bold')
                ax.axis('off')

    plt.suptitle(f"Raw index {query_img_idx} ({query_dt}) - Proposed vs. Group1 vs. Group2 [{region_key}]", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir(cfg), f"all_models_comparison_{region_key}_{query_img_idx}.png"), dpi=200, bbox_inches='tight')
    plt.show()


def main(cfg=CONFIG):
    run_unified_mse_and_significance(cfg)
    # Example: any index in the query pool (final 20%). The first call is slow
    # because DB embeddings are cached per model; later calls are fast.
    inspect_raw_index_all_models(query_img_idx=SPLIT_IDX + cfg["seq_len"] + 100, region_key="south_korea", cfg=cfg)
    # To compare selected models only:
    # inspect_raw_index_all_models(query_img_idx=SPLIT_IDX + cfg["seq_len"] + 100, region_key="south_korea",
    #                               model_names=["Proposed", "CLIP (Group1, off-the-shelf)"])


if __name__ == "__main__":
    main()
