"""Paper figures drawn from the CSVs in the run dir.

Writes PDFs to FIG_OUT_DIR (default: <runs>/figures) and PNG previews to
FIG_PREVIEW_DIR (default: <run dir>/figures_preview). Figures used in the
final paper: fig9_single_vs_video (Fig. 7), fig10_d_ablation (Fig. 8) and
fig11_pe_ablation (Fig. 9). The final Fig. 6 and Fig. 10 come from
analysis/plot_fig6.py and analysis/plot_fig10_failure.py; the versions drawn
here and the remaining panels are supplementary.
"""
import os
import re
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from rap_weather.config import CONFIG, RUNS_DIR, run_dir

RUN = run_dir(CONFIG)
OUT = os.environ.get('FIG_OUT_DIR', str(RUNS_DIR / 'figures'))
# PNG preview output directory (override with FIG_PREVIEW_DIR).
SCRATCH = os.environ.get('FIG_PREVIEW_DIR', os.path.join(RUN, 'figures_preview'))
os.makedirs(OUT, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)

# palette of the original paper figures (sampled from Total_Results /
# Patch_size_comparison / vitpe_comparison / rap_single_video_comparison)
GREENP = '#b5feb5'   # RaP-Weather
PINK   = '#febebe'   # DINOv2
PEACH  = '#fed1b2'   # ConvNeXt
YELLOWP = '#fdfac5'  # CLIP
PERI   = '#9dacfe'   # auxiliary (single-image baseline family)
LINE_GREENS = ['#a1d99b', '#74c476', '#31a354', '#006d2c']  # Dim 4/8/16/32
VITPE_Y, VITPE_G = '#faffab', '#bae4b4'
AMBER_LINE = '#c9a227'   # line-weight counterpart of the pastel yellow
PINK_LINE = '#e07070'    # line-weight counterpart of the pastel pink
PEACH_LINE = '#e6935c'   # line-weight counterpart of the pastel peach
GREEN_LINE = '#2f9e44'
EDGE = '#1a1a1a'

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    # FIG_FONT=Caladea (Cambria-metric open font, ~/.fonts/caladea) switches every label;
    # the default is DejaVu Serif.
    'font.family': 'serif', 'font.serif': [os.environ.get('FIG_FONT', 'DejaVu Serif'), 'DejaVu Serif'],
    'mathtext.fontset': 'custom' if os.environ.get('FIG_FONT') else 'dejavuserif',
    'mathtext.rm': os.environ.get('FIG_FONT', 'DejaVu Serif'),
    'mathtext.it': os.environ.get('FIG_FONT', 'DejaVu Serif') + ':italic',
    'mathtext.bf': os.environ.get('FIG_FONT', 'DejaVu Serif') + ':bold',
    # >= 8 pt for axis/tick/legend text, 7 pt for value annotations (IEEE Access: figure labels ~8-10 pt)
    'font.size': 8, 'axes.labelsize': 8.5, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.edgecolor': EDGE, 'axes.linewidth': 0.8,
    'legend.frameon': True, 'legend.framealpha': 1.0,
    'legend.edgecolor': '#333333', 'legend.fancybox': False,
    'pdf.fonttype': 42,
})

# FULLDIM=1 reads the CSVs recomputed with native-dimension baseline encoders
# instead of the D=8 compressed ones (both sets of files are kept side by side).
FULLDIM = bool(os.environ.get('FULLDIM'))
P_SIG = ((lambda r: f'{RUN}/unified_significance_fulldim_{r}.csv') if FULLDIM
         else (lambda r: f'{RUN}/unified_significance_{r}.csv'))
P_G3 = (f'{RUN}/group3_fulldim_wholetrain_crop_south_korea.csv' if FULLDIM
        else f'{RUN}/group3_wholetrain_crop_south_korea.csv')
P_SEASONAL = (f'{RUN}/seasonal_generalization_fulldim.csv' if FULLDIM
              else f'{RUN}/seasonal_generalization.csv')
print(f"[figures] FULLDIM={FULLDIM} | sig={os.path.basename(P_SIG('south_korea'))} "
      f"| g3={os.path.basename(P_G3)} | seasonal={os.path.basename(P_SEASONAL)}")

K_LIST = [1, 3, 5, 10, 50, 100]
KX = np.arange(len(K_LIST))
KLABELS = [f'Top-{k}' for k in K_LIST]


def style_axes(ax, grid_axis='y'):
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color='#cccccc', linewidth=0.6,
                linestyle='--')
    ax.tick_params(length=2.5, width=0.7)


def parse_ci(s):
    # Keep the sign so negative CI lower bounds are parsed correctly.
    lo, hi = re.findall(r'[-0-9.]+', str(s))
    return float(lo), float(hi)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', pad_inches=0.05)
    fig.savefig(os.path.join(SCRATCH, 'new_' + name + '.png'), dpi=150,
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print('saved', name)


def line_with_band(ax, means, los, his, color, label, mfc, ls='-', marker='o'):
    ax.fill_between(KX, los, his, color=color, alpha=0.22, lw=0, zorder=1)
    ax.plot(KX, means, color=color, lw=1.3, ls=ls, marker=marker, ms=3.8,
            mfc=mfc, mec=EDGE, mew=0.6, label=label, zorder=3)


# ---------------------------------------------------------------- downsampling
ds = pd.read_csv(f'{RUN}/downsampling_autocorrelation.csv')
ds = ds[ds['lag_hours'] <= 24]
fig, ax = plt.subplots(figsize=(3.5, 2.0))
ax.plot(ds['lag_hours'], ds['mean_autocorrelation'], color=GREEN_LINE, lw=1.5,
        marker='o', ms=4, mfc=GREENP, mec=EDGE, mew=0.7, zorder=3)
ax.annotate('0.830', (3, 0.8297), xytext=(8, -10), textcoords='offset points',
            fontsize=7.5, color='#333333')
ax.set_xlabel('Lag (hours)')
ax.set_ylabel('Mean autocorrelation')
ax.set_xticks(ds['lag_hours'])
style_axes(ax)
save(fig, 'downsampling_justification')

# ------------------------------------------------------------- unified G1/3
# Main-text panel: only methods that answer arbitrary regions without
# retraining (RaP + code G1 + code G3). Per-region encoders (code G2, an
# upper-bound reference) are reported in the appendix tables instead.
# Data: unified_main_significance_{region}.csv, which also has CIs and p-values
# for code G3. At total_patches the code-G3 crop is the identity, so it is the
# same model as the whole-image fine-tuned encoder.
P_MAIN = lambda r: f'{RUN}/unified_main_significance_{r}.csv'
SHORT = {
    'Proposed': 'RaP-Weather',
    'CLIP (Group1, off-the-shelf)': 'CLIP G1',
    'ConvNeXt2 (Group1, off-the-shelf)': 'ConvNeXt2 G1',
    'DINOv2 (Group1, off-the-shelf)': 'DINOv2 G1',
    'CLIP (Group3, whole-trained + crop)': 'CLIP G2',
    'ConvNeXt2 (Group3, whole-trained + crop)': 'ConvNeXt2 G2',
    'DINOv2 (Group3, whole-trained + crop)': 'DINOv2 G2',
    'CLIP (Group3, whole-image fine-tuned)': 'CLIP G2',
    'ConvNeXt2 (Group3, whole-image fine-tuned)': 'ConvNeXt2 G2',
    'DINOv2 (Group3, whole-image fine-tuned)': 'DINOv2 G2',
}
# Paper numbering: the whole-image-trained family (code Group3) is the paper's
# Group 2, so the 'G2' figure label means this family (PERI). The appendix
# figure below sets its colors directly in conds.
GROUP_FILL = {'RaP': GREENP, 'G1': PINK, 'G2': PERI}

def group_fill(label):
    for key in ('G1', 'G2'):
        if key in label:
            return GROUP_FILL[key]
    return GROUP_FILL['RaP']

def load_panel(region):
    sig = pd.read_csv(P_MAIN(region))
    t1 = sig[sig['Top-k'] == 1]
    rows = []
    for _, r in t1.iterrows():
        if r['Model'] not in SHORT:          # Pixel-NN appears only in Table 3
            continue
        rows.append((SHORT[r['Model']], r['Mean MSE'],
                     r['95% CI Low'], r['95% CI High']))
    rows.sort(key=lambda r: r[1])
    return rows

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9),
                         gridspec_kw={'width_ratios': [1, 1]})
for ax, region, title in (
        (axes[0], 'south_korea', 'Regional query (south_korea)'),
        (axes[1], 'total_patches', 'Full-image query (total_patches)')):
    rows = load_panel(region)
    y = np.arange(len(rows))[::-1]
    prop = next(r for r in rows if r[0] == 'RaP-Weather')
    ax.axvspan(prop[2], prop[3], color=GREENP, alpha=0.45, zorder=0)
    for yi, (lab, mean, lo, hi) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color='#333333', lw=0.9, zorder=2)
        ax.plot(mean, yi, 'o', ms=5.2, mfc=group_fill(lab), mec=EDGE,
                mew=0.8, zorder=3)
        ax.annotate(f'{mean:.4f}', (hi, yi), xytext=(4, 0),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=6.2, color='#333333', zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlabel('Weather-observation MSE @ Top-1')
    ax.set_title(title, fontsize=8.5, pad=4)
    ax.margins(x=0.14)
    style_axes(ax, grid_axis='x')
handles = [Line2D([], [], marker='o', ls='', mfc=c, mec=EDGE, mew=0.8, ms=5.2,
                  label=l)
           for c, l in [(GREENP, 'RaP-Weather'), (PINK, 'Group 1 · off-the-shelf'),
                        (PERI, 'Group 2 · whole-image-trained')]]
fig.legend(handles=handles, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.05), columnspacing=1.2, handletextpad=0.3)
fig.tight_layout(rect=[0, 0.06, 1, 1])
save(fig, 'fig_unified_g123')

# ----------------------------------------- code Group 3 (paper Group 2)
g3 = pd.read_csv(P_G3, index_col=0)
backbones = ['CLIP', 'DINOv2', 'ConvNeXt2']
conds = [('(G2: region-trained)', 'per-region fine-tuned', YELLOWP, 'o'),
         ('(G3: whole-trained, crop@query)', 'Group 2 · crop resized', PERI, '^'),
         ('(G3-pad/zero: whole-trained, crop@query)', 'Group 2 · crop zero-padded', PERI, 's'),
         ('(G3-pad/mean: whole-trained, crop@query)', 'Group 2 · crop mean-padded', PERI, 'D')]
FILLED = {'o', '^'}
fig, ax = plt.subplots(figsize=(3.5, 2.9))
x = np.arange(len(backbones))
offs = np.linspace(-0.24, 0.24, len(conds))
for (suffix, lab, c, mk), off in zip(conds, offs):
    vals = [g3.loc[f'{b} {suffix}', 'Top-1'] for b in backbones]
    ax.scatter(x + off, vals, marker=mk, s=26, zorder=3, label=lab,
               facecolors=(c if mk in FILLED else 'white'), edgecolors=EDGE,
               linewidths=0.8)
prop = g3.loc['Proposed', 'Top-1']
ax.axhline(prop, color=GREEN_LINE, lw=1.4, zorder=2)
# Label sits in the top margin of the axes: the data cluster near the reference
# line (0.98-1.17), so a label right above the line would overlap the DINOv2 and
# ConvNeXt2 markers.
ax.annotate(f'RaP-Weather (no crop)  {prop:.3f}', (0.99, 0.965),
            xycoords='axes fraction', ha='right', va='top', fontsize=7,
            color=GREEN_LINE)
ax.set_xticks(x)
ax.set_xticklabels(backbones)
ax.set_ylabel('Weather-obs. MSE')
ax.set_ylim(0.93, 1.32)
style_axes(ax)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.46), ncol=2,
          columnspacing=0.9, handletextpad=0.3)
fig.tight_layout(rect=[0, 0.12, 1, 1])
save(fig, 'fig_group3_wholetrain_crop')

# --------------------------------------------- D ablation (lines w/ CI bands)
da = pd.read_csv(f'{RUN}/d_ablation_results.csv')
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for color, (_, row) in zip(LINE_GREENS, da.iterrows()):
    D = int(row['D'])
    means = [row[f'Top-{k} MSE'] for k in K_LIST]
    los, his = zip(*[parse_ci(row[f'Top-{k} 95% CI']) for k in K_LIST])
    line_with_band(ax, means, los, his, color, f'Dim: {D}', color)
ax.set_xticks(KX)
ax.set_xticklabels(KLABELS, fontsize=8)
ax.set_xlabel('Retrieval Performance @ Top-$k$')
ax.set_ylabel('Weather-observation MSE')
ax.set_xlim(-0.25, 5.25)
ax.set_ylim(0.85, 1.68)
style_axes(ax)
ax.legend(title='Patch Emb Size', title_fontsize=8, loc='center left',
          borderpad=0.35, handletextpad=0.4, handlelength=1.6)
fig.tight_layout()
save(fig, 'fig10_d_ablation')

# -------------------------------------------- PE ablation (lines w/ CI bands)
pe = pd.read_csv(f'{RUN}/pe_ablation_results.csv')
fig, ax = plt.subplots(figsize=(3.5, 2.4))
order = [('w/ Positional Encoding', AMBER_LINE, VITPE_Y, '--', 's',
          'w/ Positional Encoding'),
         ('w/o Positional Encoding (Proposed)', GREEN_LINE, VITPE_G, '-', 'o',
          'w/o Positional Encoding (Ours)')]
for variant, color, mfc, ls, marker, lab in order:
    row = pe[pe['Variant'] == variant].iloc[0]
    means = [row[f'Top-{k} MSE'] for k in K_LIST]
    los, his = zip(*[parse_ci(row[f'Top-{k} 95% CI']) for k in K_LIST])
    line_with_band(ax, means, los, his, color, lab, mfc, ls=ls, marker=marker)
ax.set_xticks(KX)
ax.set_xticklabels(KLABELS, fontsize=8)
ax.set_xlabel('Retrieval Performance @ Top-$k$')
ax.set_ylabel('Weather-observation MSE')
ax.set_xlim(-0.25, 5.25)
ax.set_ylim(0.97, 1.14)
style_axes(ax)
ax.legend(loc='upper left', borderpad=0.35, handlelength=1.6,
          handletextpad=0.4)
fig.tight_layout()
save(fig, 'fig11_pe_ablation')

# ------------------- seasonal JJA (lines w/ code Group 3 comparison)
# Compared against the main-text trained baseline, code G3 (whole-trained + crop).
sea = pd.read_csv(f'{RUN}/seasonal_generalization_main.csv')
sk = sea[sea['Region'] == 'south_korea']
# Match by prefix: the season label inside the column name may differ between runs.
JJA = next(c for c in sea.columns if c.startswith('JJA') and c.endswith(' MSE'))
models = [('CLIP (Group3, whole-trained + crop)', 'CLIP G2',
           AMBER_LINE, YELLOWP, '--', 's'),
          ('DINOv2 (Group3, whole-trained + crop)', 'DINOv2 G2',
           PINK_LINE, PINK, '--', '^'),
          ('ConvNeXt2 (Group3, whole-trained + crop)', 'ConvNeXt2 G2',
           PEACH_LINE, PEACH, '--', 'D'),
          ('Proposed', 'RaP-Weather (Ours)', GREEN_LINE, GREENP, '-', 'o')]
fig, ax = plt.subplots(figsize=(3.5, 2.6))
all_vals = []
for model, lab, color, mfc, ls, marker in models:
    d = sk[sk['Model'] == model].sort_values('Top-k')
    all_vals.extend(d[JJA].values)
    ax.plot(KX, d[JJA].values, color=color, lw=1.3, ls=ls, marker=marker,
            ms=3.8, mfc=mfc, mec=EDGE, mew=0.6, label=lab, zorder=3)
ax.set_xticks(KX)
ax.set_xticklabels(KLABELS, fontsize=8)
ax.set_xlabel('Retrieval Performance @ Top-$k$ (JJA)')
ax.set_ylabel('Weather-observation MSE')
ax.set_xlim(-0.25, 5.25)
ax.set_ylim(min(all_vals) - 0.035, max(all_vals) + 0.035)
style_axes(ax)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
          borderpad=0.35, columnspacing=0.8, handletextpad=0.4,
          handlelength=1.6)
fig.tight_layout()
save(fig, 'fig18_seasonal_jja')

# ------------------- single-image vs video model (lines w/ CI bands + values)
sv_path = f'{RUN}/single_vs_proposed_current_protocol_south_korea.csv'
if os.path.exists(sv_path):
    sv = pd.read_csv(sv_path).sort_values('Top-k')
    # Proposed line: the published numbers (unified_significance CSV) so the
    # figure matches Table 7 exactly; the re-run only backs the paired test.
    us = pd.read_csv(P_SIG('south_korea'))
    us = us[us['Model'] == 'Proposed'].sort_values('Top-k')
    PERI_LINE = '#5b6fd6'
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    line_with_band(ax, sv['RaP-Single MSE'], sv['Single CI Low'],
                   sv['Single CI High'], PERI_LINE, 'RaP-Single Image', PERI,
                   ls='--', marker='s')
    line_with_band(ax, us['Mean MSE'].values, us['95% CI Low'].values,
                   us['95% CI High'].values, GREEN_LINE, 'RaP-Weather (Ours)',
                   GREENP)
    for x, (s, slo, shi, m, plo) in enumerate(zip(
            sv['RaP-Single MSE'], sv['Single CI Low'], sv['Single CI High'],
            us['Mean MSE'].values, us['95% CI Low'].values)):
        ax.annotate(f'{s:.3f}', (x, shi), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=7,
                    color='#333333', zorder=4)
        ax.annotate(f'{m:.3f}', (x, plo), xytext=(0, -9),
                    textcoords='offset points', ha='center', fontsize=7,
                    color='#333333', zorder=4)
    ax.set_xticks(KX)
    ax.set_xticklabels(KLABELS, fontsize=8)
    ax.set_xlabel('Retrieval Performance @ Top-$k$')
    ax.set_ylabel('Weather-observation MSE')
    ax.set_xlim(-0.35, 5.35)
    lo = us['95% CI Low'].min() - 0.045
    hi = sv['Single CI High'].max() + 0.045
    ax.set_ylim(lo, hi)
    style_axes(ax)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
              borderpad=0.35, columnspacing=0.8, handletextpad=0.4,
              handlelength=1.6)
    fig.tight_layout()
    save(fig, 'fig9_single_vs_video')
else:
    print('single_vs_proposed CSV not found - skipping fig9')

# ------------------------------------------------------------- failure cases
if os.path.exists(f'{SCRATCH}/failcase.npz'):
    d = np.load(f'{SCRATCH}/failcase.npz', allow_pickle=True)
    worst = d['worst'][:3]
    per_q = d['per_q']
    h5q, h5r = d['h5q'][:3], d['h5r'][:3]
    qdts, rdts = d['qdts'][:3], d['rdts'][:3]

    def fmt(dt):
        s = str(dt)
        return f'{s[0:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}'

    with h5py.File(CONFIG['paths']['h5'], 'r') as f:
        qimgs = [f['images'][int(i)] for i in h5q]
        rimgs = [f['images'][int(i)] for i in h5r]

    BOX = (96, 80, 48, 64)
    fig, axes2 = plt.subplots(3, 2, figsize=(3.5, 5.6))
    for r in range(3):
        mse = per_q[int(worst[r])]
        for c, (img, dt) in enumerate([(qimgs[r], qdts[r]), (rimgs[r], rdts[r])]):
            ax = axes2[r, c]
            ax.imshow(np.transpose(img, (1, 2, 0)))
            ax.add_patch(mpatches.Rectangle((BOX[0], BOX[1]), BOX[2], BOX[3],
                         fill=False, edgecolor='#E8442B', linewidth=1.4))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_edgecolor(EDGE)
                s.set_linewidth(0.7)
            title = (f'Query · {fmt(dt)}' if c == 0
                     else f'Top-1 · {fmt(dt)} · MSE {mse:.2f}')
            ax.set_title(title, fontsize=7, pad=2.5)
    fig.subplots_adjust(wspace=0.04, hspace=0.15, left=0.01, right=0.99,
                        top=0.965, bottom=0.005)
    save(fig, 'fig20_failure_cases')
else:
    print('failcase.npz not found - skipping fig20')

print('all done')
