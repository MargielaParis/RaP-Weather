"""Fig. 2 (visual change over one 3-hour step) for the two timestamps shown in the paper.
FIG_FONT selects the label font (default Times New Roman)."""
import os, sys
import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT = os.environ.get("FIG_FONT", "Times New Roman")
plt.rcParams.update({"font.family": "serif", "font.serif": [FONT, "DejaVu Serif"], "pdf.fonttype": 42,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
from _paths import FIGURES_DIR, H5
DTS = ["201102280000", "201102280300"]
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
out = sys.argv[1] if len(sys.argv) > 1 else str(FIGURES_DIR / "dif_bit_3h.pdf")
with h5py.File(H5, "r") as f:
    dts = [d.decode() for d in f["datetimes"][:]]
    idx = {d: i for i, d in enumerate(dts)}
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.4))
    for ax, dt in zip(axes, DTS):
        ax.imshow(np.transpose(f["images"][idx[dt]], (1, 2, 0)), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.6)
        ax.set_title(f"{dt[0:4]}/{dt[4:6]}/{dt[6:8]}\n{dt[8:10]}:{dt[10:12]}", fontsize=9.5, pad=3)
    fig.subplots_adjust(wspace=0.22, left=0.01, right=0.99, top=0.85, bottom=0.01)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03, dpi=300)  # >=300 dpi for the embedded satellite rasters (IEEE Access IV-E)
print("wrote", out, "font", FONT)
