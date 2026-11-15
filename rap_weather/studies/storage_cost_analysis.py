"""Storage cost of precomputed patch embeddings.

Precomputing and storing per-patch embeddings removes the need to re-encode
raw images at query time, at the cost of disk space. For the actual database
size, this computes (1) raw image storage, (2) patch embedding storage per
embedding size D, and (3) the storage/compute trade-off against the
crop-then-recompute approach.
"""
import os

import h5py
import pandas as pd

from ..config import CONFIG, run_dir

def compute_storage_cost(cfg=CONFIG, split_idx=None):
    with h5py.File(cfg["paths"]["h5"], "r") as f:
        images = f["images"]
        total_samples = len(images)
        raw_shape = images.shape[1:]
        raw_dtype = images.dtype

    split_idx = split_idx if split_idx is not None else int(total_samples * cfg["train_val_split"])
    n_db = split_idx - cfg["seq_len"]
    n_patches = 196
    bytes_per_float32 = 4

    # 1) Uncompressed raw image size, using the actual HDF5 dtype
    raw_image_bytes_per_sample = int(pd.Series(raw_shape).prod()) * raw_dtype.itemsize
    raw_image_total_gb = (raw_image_bytes_per_sample * n_db) / (1024 ** 3)

    # 2) Patch embedding storage per D (same D values as the Fig. 8 ablation)
    d_values = [4, 8, 16, 32]
    storage_rows = []
    for D in d_values:
        emb_bytes_per_sample = n_patches * D * bytes_per_float32
        emb_total_gb = (emb_bytes_per_sample * n_db) / (1024 ** 3)
        storage_rows.append({
            "D": D,
            "Raw image dtype": str(raw_dtype),
            "Raw bytes/sample": raw_image_bytes_per_sample,
            "Bytes/sample": emb_bytes_per_sample,
            "Total DB size (GB)": emb_total_gb,
            "vs. raw image storage": f"{emb_total_gb / raw_image_total_gb:.2%}",
            "Compression ratio (raw / emb)": raw_image_bytes_per_sample / emb_bytes_per_sample,
        })

    storage_df = pd.DataFrame(storage_rows)
    print(f"DB samples: {n_db:,}")
    print(f"Raw image storage ({raw_dtype}, before HDF5 compression): {raw_image_total_gb:.2f} GB\n")
    print(storage_df.to_string(index=False))

    # 3) Compute trade-off: crop and re-encode per region vs. index into stored patch embeddings
    #    -> number of ViT forward passes per query
    n_regions_example = 4  # e.g. south_korea, low_left, total_patches used in the paper
    print(f"\nCompute comparison (one query, {n_regions_example} regions)")
    print(f"   - Crop-then-recompute: one ViT forward pass per region -> {n_regions_example} encodings")
    print(f"   - No-Crop-No-Recompute (ours): encode once, then index patches -> 1 encoding + O(1) indexing")

    out_path = os.path.join(run_dir(cfg), "storage_cost_analysis.csv")
    storage_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print(f"\nSummary sentence (D={cfg['patch_emb_size']}):")
    row = storage_df[storage_df["D"] == cfg["patch_emb_size"]].iloc[0]
    print(f'   "At D={cfg["patch_emb_size"]}, storing precomputed patch embeddings for the '
          f'full {n_db:,}-sample database requires {row["Total DB size (GB)"]:.2f} GB '
          f'({row["vs. raw image storage"]} of the raw image storage), while avoiding '
          f'repeated ViT forward passes at query time for each region of interest."')
    return storage_df


if __name__ == "__main__":
    compute_storage_cost()
