"""Figure 6 (Top-1 weather-observation MSE, regional vs full-image) from primary_main_stats.csv.

Sized for print. Default is a single-column stacked layout (3.4 x 3.3 in at \\linewidth) so the
figure sits in a text column and does not compete with the full-width Table 4 for the top of the
page; text is 8-9.5 pt, giving roughly 8 pt on the page. Called by
statistical_audit.py and runnable on its own.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from _paths import FIGURES_DIR, STATS_DIR as HERE

LABELS = ["RaP-Weather", "CLIP G1", "ConvNeXt2 G1", "DINOv2 G1", "CLIP G2", "ConvNeXt2 G2", "DINOv2 G2"]
COLORS = ["#138a72", "#4daf4a", "#4daf4a", "#4daf4a", "#e41a1c", "#e41a1c", "#e41a1c"]


def plot_main_top1(frame: pd.DataFrame, output: Path, stacked: bool = True) -> None:
    """stacked=True: two panels one above the other for a single-column figure
    (3.4 x 3.3 in, placed at \\linewidth). stacked=False: side by side for figure*."""
    # FIG_FONT (e.g. Caladea, the Cambria-metric open font in ~/.fonts/caladea) selects the label font.
    font = os.environ.get("FIG_FONT", "Times New Roman")
    plt.rcParams.update({"font.family": [font, "DejaVu Sans"], "pdf.fonttype": 42,
                         "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    scopes = ("south_korea", "total_patches")
    if stacked:
        fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.3), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for ax, scope in zip(axes, scopes):
        data = frame[(frame["Scope"] == scope) & (frame["Top-k"] == 1)].copy().iloc[:7]
        y = data["Mean"].to_numpy(float)
        low = data["95% CI Low"].to_numpy(float)
        high = data["95% CI High"].to_numpy(float)
        x = np.arange(len(data))
        if stacked:
            # horizontal dot plot: model names on the y axis read without rotation in a narrow column
            pos = x[::-1]
            ax.errorbar(y, pos, xerr=np.vstack([y - low, high - y]), fmt="none", ecolor="#555555", capsize=3, lw=1.2)
            ax.scatter(y, pos, c=COLORS, s=34, zorder=3)
            ax.set_yticks(pos, LABELS)
            ax.set_xlabel("Top-1 weather-observation MSE")
            ax.grid(axis="x", alpha=0.25)
            ax.set_ylim(-0.7, len(data) - 0.3)
        else:
            ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), fmt="none", ecolor="#555555", capsize=3, lw=1.2)
            ax.scatter(x, y, c=COLORS, s=34, zorder=3)
            ax.set_xticks(x, LABELS, rotation=42, ha="right")
            ax.set_ylabel("Top-1 weather-observation MSE")
            ax.grid(axis="y", alpha=0.25)
        ax.set_title("Regional query" if scope == "south_korea" else "Full-image query")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    frame = pd.read_csv(HERE / "primary_main_stats.csv")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_main_top1(frame, FIGURES_DIR / "fig_unified_g123.pdf")
    print("wrote", FIGURES_DIR / "fig_unified_g123.pdf")
