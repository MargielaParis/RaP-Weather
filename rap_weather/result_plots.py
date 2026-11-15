"""Result plots, read from the CSVs saved by the evaluation modules."""
import os

import pandas as pd
import matplotlib.pyplot as plt

from .config import CONFIG, run_dir


def main(cfg=CONFIG):
    mse_csv = os.path.join(run_dir(cfg), "mse_comparison_south_korea_D" + str(cfg["patch_emb_size"]) + ".csv")
    if os.path.exists(mse_csv):
        mse_df = pd.read_csv(mse_csv, index_col=0)
        mse_df.columns = [int(c.replace("Top-", "")) for c in mse_df.columns]
        fig, ax = plt.subplots(figsize=(12, 6))
        mse_df.T.plot(kind='bar', ax=ax, edgecolor='black', linewidth=1.0, zorder=3)
        ax.set_title("Retrieval Performance vs. Ground-Truth Weather (South Korea)", fontsize=16)
        ax.set_xlabel("Top-k Retrieved"); ax.set_ylabel("Average Ground Truth MSE (Lower is Better)")
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        plt.tight_layout(); plt.show()
    else:
        print(f"Not found: {mse_csv}; run eval_mse_retrieval first.")

    table1_csv = os.path.join(run_dir(cfg), "table1_quality_comparison.csv")
    if os.path.exists(table1_csv):
        t1_df = pd.read_csv(table1_csv)
        print(t1_df)
    else:
        print(f"Not found: {table1_csv}; run eval_image_quality first.")


if __name__ == "__main__":
    main()
