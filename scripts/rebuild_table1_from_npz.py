"""Rebuild the combined Table 1 CSVs from saved per-query NPZ artifacts."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


TOP_K_LIST = (1, 3, 5, 10, 50, 100)
METRICS = ("LPIPS", "PSNR", "SSIM")


def load_raw(path):
    raw = {}
    with np.load(path) as archive:
        for key in archive.files:
            model, metric, top_k = key.split("||")
            raw.setdefault(model, {}).setdefault(metric, {})[int(top_k)] = archive[key]
    return raw


def bootstrap_ci(values, n_boot=2000, ci=95, seed=42):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    tail = (100 - ci) / 2
    return values.mean(), np.percentile(means, tail), np.percentile(means, 100 - tail)


def summarize_region(region, raw):
    rows = []
    for model, values in raw.items():
        row = {
            "Scenario": region,
            "Model": model,
            "N": len(values["LPIPS"][TOP_K_LIST[0]]),
        }
        for metric in METRICS:
            for top_k in TOP_K_LIST:
                row[f"{metric}@{top_k}"] = float(np.mean(values[metric][top_k]))
        rows.append(row)
    return rows


def significance_region(region, raw):
    rows = []
    baseline = raw.get("RaP-Weather")
    for model, values in raw.items():
        for metric in METRICS:
            for top_k in TOP_K_LIST:
                sample = np.asarray(values[metric][top_k], dtype=np.float64)
                mean, low, high = bootstrap_ci(sample)
                row = {
                    "Scenario": region,
                    "Metric": metric,
                    "Model": model,
                    "Top-k": top_k,
                    "Mean": mean,
                    "95% CI Low": low,
                    "95% CI High": high,
                    "N": len(sample),
                }
                if baseline is not None and model != "RaP-Weather":
                    base = np.asarray(baseline[metric][top_k], dtype=np.float64)
                    if len(base) == len(sample) and len(sample) > 1:
                        try:
                            p_value = stats.wilcoxon(base, sample).pvalue
                        except ValueError:
                            p_value = np.nan
                        row["Wilcoxon p (vs. RaP-Weather)"] = p_value
                        row["Significant (p<0.05)"] = bool(p_value < 0.05) if not np.isnan(p_value) else False
                        row["Better than RaP"] = bool(
                            sample.mean() < base.mean()
                            if metric == "LPIPS"
                            else sample.mean() > base.mean()
                        )
                rows.append(row)
    return rows


def main():
    from rap_weather.config import CONFIG, run_dir
    default_run = Path(run_dir(CONFIG))
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run)
    args = parser.parse_args()

    rows = []
    significance = []
    sources = sorted(args.run_dir.glob("table1_group13_perquery_*.npz"))
    sources = [path for path in sources if "_limit" not in path.stem]
    if not sources:
        raise FileNotFoundError(f"No Table 1 per-query NPZ files in {args.run_dir}")

    for path in sources:
        region = path.stem.removeprefix("table1_group13_perquery_")
        raw = load_raw(path)
        rows.extend(summarize_region(region, raw))
        significance.extend(significance_region(region, raw))

    table_path = args.run_dir / "table1_group13.csv"
    sig_path = args.run_dir / "table1_group13_significance.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    pd.DataFrame(significance).to_csv(sig_path, index=False)
    print(f"saved {len(rows)} rows: {table_path}")
    print(f"saved {len(significance)} rows: {sig_path}")


if __name__ == "__main__":
    main()
