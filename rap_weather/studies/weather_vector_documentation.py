"""Documentation of the meteorological observation vector (source,
dimensionality, variables, construction) used in Figures 6 and 7.

Loads the vectors from `coms_with_weather_dataset.pkl` and computes the
dimensionality, basic statistics, and missing values. Filling `variable_names`
with the actual variable names (e.g. per-station KMA temperature, humidity,
pressure, wind speed) yields a table ready for Section III or the Appendix.
"""
import os
import pickle

import numpy as np
import pandas as pd

from ..config import CONFIG, run_dir
from ..models import clean_weather_vectors


def document_weather_vector(cfg=CONFIG):
    with open(cfg["paths"]["weather_pkl"], "rb") as f:
        gt_data = pickle.load(f)

    raw_vecs = np.array([item[2] for item in gt_data])
    clean_vecs, sentinel_mask = clean_weather_vectors(raw_vecs)
    sample_dt = gt_data[0][0]
    vec_dim = raw_vecs.shape[1]

    print(f"Source file: {cfg['paths']['weather_pkl']}")
    print(f"Vector dimension: {vec_dim}")
    print(f"Samples: {len(gt_data)} (e.g. {sample_dt})")
    print(f"Samples with NaN: {np.isnan(raw_vecs).any(axis=1).sum()} / {len(raw_vecs)}")
    print(f"Sentinel cleaning: {int(sentinel_mask.sum())} values, "
          f"{int(sentinel_mask.any(axis=1).sum())} samples, "
          f"{int(sentinel_mask.any(axis=0).sum())} columns")

    # TODO: fill in the weather variable names in vector order,
    # following the data generation script (e.g. the KMA API parser).
    variable_names = [f"var_{i}" for i in range(vec_dim)]  # e.g. ["temperature", "humidity", "pressure", ...]

    stats_df = pd.DataFrame({
        "variable": variable_names,
        "mean": clean_vecs.mean(axis=0),
        "std": clean_vecs.std(axis=0),
        "min": clean_vecs.min(axis=0),
        "max": clean_vecs.max(axis=0),
        "nan_count": np.isnan(raw_vecs).sum(axis=0),
        "sentinel_count": sentinel_mask.sum(axis=0),
    })
    print("\nPer-variable statistics (fill in variable_names to complete the table):")
    print(stats_df.to_string(index=False))

    out_path = os.path.join(run_dir(cfg), "weather_vector_documentation.csv")
    stats_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Auto-generated summary sentence
    print(f"\nSummary sentence:")
    print(f'   "The meteorological observation vector is a {vec_dim}-dimensional vector '
          f'derived from {os.path.basename(cfg["paths"]["weather_pkl"])}, '
          f'covering {len(gt_data)} timestamps. Each dimension is standardized '
          f'(StandardScaler, zero mean unit variance) before use in MSE computation."')
    return stats_df


if __name__ == "__main__":
    document_weather_vector()
