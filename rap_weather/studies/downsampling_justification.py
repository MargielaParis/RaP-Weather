"""Empirical justification for downsampling satellite observations from
15-minute to 3-hour intervals.

The pipeline only has the already-downsampled 3-hour data
(`coms_25744_224.hdf5`), so a direct comparison with the native 15-minute
data is not possible. Instead this module provides:
1. What can be measured on the 3-hour data: lagged autocorrelation of the
   weather vectors.
2. What requires the native 15-minute data: a scaffold that measures the
   actual information loss (runs once the native file path is given).
"""
import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config import CONFIG, run_dir
from ..models import get_weather_map_and_scaler


# =========================================================
# Measurable on the 3-hour data: autocorrelation and rate of change
# =========================================================
def lagged_autocorr(series, max_lag):
    """Per-dimension autocorrelation at lag k (in units of 3h steps), averaged
    across dimensions. High values at small k = weather state changes slowly
    relative to the 3h sampling interval (supports the downsampling choice for
    a RETRIEVAL task, though NOT necessarily for fine-grained forecasting)."""
    T, D = series.shape
    result = []
    for lag in range(1, max_lag + 1):
        a, b = series[:-lag], series[lag:]
        corr = np.array([np.corrcoef(a[:, d], b[:, d])[0, 1] for d in range(D)])
        result.append(np.nanmean(corr))
    return np.array(result)


def run_autocorrelation_analysis(cfg=CONFIG, max_lag_steps=20):
    weather_map, _ = get_weather_map_and_scaler(cfg)
    sorted_dts = sorted(weather_map.keys())
    vec_series = np.stack([weather_map[dt] for dt in sorted_dts])  # (T, dim), already standardized

    acf = lagged_autocorr(vec_series, max_lag_steps)
    lags_hours = np.arange(1, max_lag_steps + 1) * 3

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(lags_hours, acf, marker='o')
    axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='ACF = 0.5')
    axes[0].set_xlabel("Lag (hours)"); axes[0].set_ylabel("Mean autocorrelation across weather-vector dims")
    axes[0].set_title("Weather-state persistence at 3h sampling"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Typical magnitude of change per 3h step vs. per larger gaps (rate-of-change distribution)
    diffs_3h = np.linalg.norm(vec_series[1:] - vec_series[:-1], axis=1)
    print(f"Change per 3h step (L2 norm, standardized vectors): "
          f"mean={diffs_3h.mean():.4f}, median={np.median(diffs_3h):.4f}, std={diffs_3h.std():.4f}")
    half_life_idx = np.argmax(acf < 0.5) if (acf < 0.5).any() else None
    if half_life_idx is not None:
        print(f"Autocorrelation drops below 0.5 after about {lags_hours[half_life_idx]} hours "
              f"(lag={half_life_idx+1} steps)")
        print(f"   -> correlation is retained on average for up to {half_life_idx+1} 3h steps "
              f"(suggests 3h resolution does not cause abrupt information loss for retrieval/similarity)")
    else:
        print("Autocorrelation stays above 0.5 over the measured lag range (very slow change)")

    axes[1].hist(diffs_3h, bins=40, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel("L2 norm of weather-vector change per 3h step"); axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of 3h-step weather change magnitude"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir(cfg), "downsampling_justification.png"), dpi=300)
    plt.show()

    acf_df = pd.DataFrame({"lag_hours": lags_hours, "mean_autocorrelation": acf})
    acf_df.to_csv(os.path.join(run_dir(cfg), "downsampling_autocorrelation.csv"), index=False)
    print(f"Saved: downsampling_autocorrelation.csv, downsampling_justification.png")
    return acf_df


# =========================================================
# Scaffold: runs only when native 15-minute data is available (direct information-loss measurement)
# =========================================================
def measure_downsampling_info_loss(native_h5_path, native_dt_key="datetimes",
                                     native_value_key="images", factor=12):
    """Compare native 15-minute data with its downsampled version and report
    the fraction of variance retained.
    factor=12: 15 min * 12 = 180 min = 3 h.

    Without native data, use only the autocorrelation analysis above
    (an approximate, not a complete, substitute).
    """
    if not os.path.exists(native_h5_path):
        print(f"File not found: {native_h5_path}. Run this scaffold once native 15-minute data is available.")
        return None

    with h5py.File(native_h5_path, "r") as f:
        native_vals = f[native_value_key][:]  # (T_native, ...)

    T_native = native_vals.shape[0]
    T_native_trimmed = (T_native // factor) * factor
    native_vals = native_vals[:T_native_trimmed]

    # Downsampling: keep the last frame of each 3h window (same as the current pipeline)
    downsampled_last = native_vals[factor - 1::factor]
    # Alternative: window mean
    downsampled_mean = native_vals.reshape(-1, factor, *native_vals.shape[1:]).mean(axis=1)

    native_var = native_vals.reshape(len(native_vals), -1).var(axis=0).mean()
    last_var = downsampled_last.reshape(len(downsampled_last), -1).var(axis=0).mean()
    mean_var = downsampled_mean.reshape(len(downsampled_mean), -1).var(axis=0).mean()

    print(f"Native (15 min) variance: {native_var:.6f}")
    print(f"Variance retained, downsampled (last frame, current method): {last_var/native_var:.2%}")
    print(f"Variance retained, downsampled (window mean): {mean_var/native_var:.2%}")
    return {"native_var": native_var, "last_frame_retained": last_var / native_var,
            "mean_pooled_retained": mean_var / native_var}


if __name__ == "__main__":
    run_autocorrelation_analysis()
    # Example, given a native 15-minute file:
    # result = measure_downsampling_info_loss("../data/coms_native_15min.hdf5")
    print("To measure the actual loss, call measure_downsampling_info_loss() with a native 15-minute HDF5 file. "
          "Without it, only the autocorrelation analysis above provides approximate evidence.")
