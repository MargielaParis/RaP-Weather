"""Fig. 10 (failure cases): the three query/Top-1 pairs shown in the paper.

Timestamps and Top-1 MSE values are the three south_korea queries with the largest Top-1
observation MSE, as listed in the paper. Layout, box, colours and
sizes follow regen_paper_figures.py. FIG_FONT selects the label font (default Times New Roman).
"""
import os, sys
import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FONT = os.environ.get("FIG_FONT", "Times New Roman")
plt.rcParams.update({"font.family": "serif", "font.serif": [FONT, "DejaVu Serif"], "pdf.fonttype": 42,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
EDGE = "#1a1a1a"
from _paths import FIGURES_DIR, H5
CASES = [("201910101500", "201204030300", 5.30),
         ("201812310900", "201203162100", 5.02),
         ("201812091800", "201211260600", 4.21)]
BOX = (96, 80, 48, 64)  # south_korea region (SK_1 + 14 offset)

def fmt(s): return f"{s[0:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
out = sys.argv[1] if len(sys.argv) > 1 else str(FIGURES_DIR / "fig20_failure_cases.pdf")
with h5py.File(H5, "r") as f:
    dts = [d.decode() for d in f["datetimes"][:]]
    idx = {d: i for i, d in enumerate(dts)}
    fig, axes = plt.subplots(3, 2, figsize=(3.5, 5.6))
    for r, (q, t, mse) in enumerate(CASES):
        for c, dt in enumerate((q, t)):
            ax = axes[r, c]
            ax.imshow(np.transpose(f["images"][idx[dt]], (1, 2, 0)), interpolation="nearest")
            ax.add_patch(mpatches.Rectangle((BOX[0], BOX[1]), BOX[2], BOX[3], fill=False, edgecolor="#E8442B", linewidth=1.4))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_edgecolor(EDGE); s.set_linewidth(0.7)
            ax.set_title(f"Query · {fmt(dt)}" if c == 0 else f"Top-1 · {fmt(dt)} · MSE {mse:.2f}", fontsize=8, pad=2.5)
    fig.subplots_adjust(wspace=0.04, hspace=0.15, left=0.01, right=0.99, top=0.965, bottom=0.005)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=300)  # >=300 dpi for the embedded satellite rasters (IEEE Access IV-E)
print("wrote", out, "font", FONT)
