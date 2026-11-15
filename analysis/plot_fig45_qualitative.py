"""Fig. 4 / Fig. 5 (qualitative retrieval) with legible labels (Times New Roman Bold 6.2 pt, model names on the left).
Query and retrieved timestamps are those of the retrieval run shown in the paper; box = south_korea_wide with the +41 offset used by that run;
images come from coms_25744_224.hdf5. Output: <runs>/figures/lingling_cropped_0.pdf and mitag_full_1.pdf."""
import sys
import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["font.weight"] = "bold"  # all labels bold; the retrieved analog (Bolaven / Malakas) is additionally 5% larger
from matplotlib.patches import Rectangle

from _paths import FIGURES_DIR, H5
f = h5py.File(H5, "r")
dts = [d.decode() for d in f["datetimes"][:]]
idx = {d: i for i, d in enumerate(dts)}
SK_BOX = (96 + 0, 64, 48, 64)  # placeholder, replaced below from regions.py logic
# box = the region that retrieval run used: south_korea_wide 4x4 block with a +41 offset
# (rap_weather/regions.py uses +14). It matches the square box drawn in the
# paper figure, which sits over the typhoon south of the peninsula.
inds = [i + 41 for i in [62, 63, 64, 65, 76, 77, 78, 79, 90, 91, 92, 93, 104, 105, 106, 107]]
rows = [i // 14 for i in inds]; cols = [i % 14 for i in inds]
SK_BOX = (min(cols) * 16, min(rows) * 16, (max(cols) - min(cols) + 1) * 16, (max(rows) - min(rows) + 1) * 16)

def key(s):  # "2012/08/26 18:00" -> "201208261800"
    return s.replace("/", "").replace(" ", "").replace(":", "")

def img(s):
    return np.transpose(f["images"][idx[key(s)]], (1, 2, 0))

FIGS = {
 "lingling": dict(query="2019/09/06 12:00", title="Query (Cropped)", box=SK_BOX, rows=[
   ("RaP-Weather", ["2012/08/26 18:00|(Bolaven)", "2012/08/26 15:00|(Bolaven)", "2012/08/26 12:00|(Bolaven)", "2014/07/31 12:00", "2014/07/31 15:00"]),
   ("DINOv2",      ["2013/10/06 18:00", "2012/09/15 21:00", "2011/11/04 12:00", "2012/09/15 15:00", "2011/10/13 21:00"]),
   ("CLIP",        ["2012/09/16 15:00", "2014/05/10 21:00", "2013/08/21 15:00", "2014/08/01 12:00", "2015/07/11 18:00"]),
   ("ConvNeXt",    ["2011/08/05 15:00", "2012/08/08 15:00", "2015/07/06 18:00", "2011/06/24 15:00", "2012/07/17 12:00"]),
 ]),
 "mitag": dict(query="2019/09/29 15:00", title="Query (Full)", box=None, rows=[
   ("RaP-Weather", ["2016/10/03 09:00", "2013/09/20 18:00", "2017/09/11 18:00", "2016/09/17 03:00|(Malakas)", "2014/10/10 03:00"]),
   ("DINOv2",      ["2014/10/09 12:00", "2013/10/12 12:00", "2014/10/02 12:00", "2012/10/14 12:00", "2013/11/02 12:00"]),
   ("CLIP",        ["2016/10/18 12:00", "2015/10/20 12:00", "2015/09/25 15:00", "2014/10/10 12:00", "2015/09/26 18:00"]),
   ("ConvNeXt",    ["2014/07/06 12:00", "2014/10/28 09:00", "2013/09/20 15:00", "2015/09/26 18:00", "2012/11/03 21:00"]),
 ]),
}

def draw(name, out, width_in=3.4, font=6.2):
    spec = FIGS[name]
    nrow, ncol = 4, 6
    tile = (width_in - 0.06 * (ncol - 1)) / ncol
    fig_h = nrow * tile + nrow * 0.20 + 0.08
    fig, axes = plt.subplots(nrow, ncol, figsize=(width_in, fig_h))
    plt.subplots_adjust(left=0.10, right=1, bottom=0.01, top=0.885, wspace=0.12, hspace=0.50)
    q = img(spec["query"])
    for r, (method, hits) in enumerate(spec["rows"]):
        for c in range(ncol):
            ax = axes[r, c]; ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.imshow(q, interpolation="none")
                d, t = spec["query"].split(" ")
                lab = f"{spec['title'].replace(' ', chr(10))}\n{d}\n{t}" if r == 0 else f"{d}\n{t}"
                ax.set_title(lab, fontsize=font, pad=2)
                ax.text(-0.08, 0.5, method, rotation=90, va="center", ha="right", fontsize=font + 1.4, transform=ax.transAxes, clip_on=False)
            else:
                s = hits[c - 1]
                ax.imshow(img(s.split("|")[0]), interpolation="none")
                d, t = s.split("|")[0].split(" ")
                extra = ("\n" + s.split("|")[1]) if "|" in s else ""
                ax.set_title(f"{d}\n{t}{extra}", fontsize=(font * 1.05 if "|" in s else font), fontweight="bold", pad=2)
            if spec["box"] is not None:
                x, y, w, h = spec["box"]
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", lw=1.0))
            else:
                for sp in ax.spines.values(): sp.set_edgecolor("red"); sp.set_linewidth(1.0)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out, "size", fig.get_size_inches())

if __name__ == "__main__":
    missing = [s for spec in FIGS.values() for _, hits in spec["rows"] for s in [spec["query"]] + [h.split("|")[0] for h in hits] if key(s) not in idx]
    print("missing timestamps:", missing)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    draw("lingling", str(FIGURES_DIR / "lingling_cropped_0"))
    draw("mitag", str(FIGURES_DIR / "mitag_full_1"))
