"""LaTeX tables and result sentences for the paper, generated from the
statistics CSVs so every reported number traces back to one source.

Inputs (stats dir): primary_main_stats.csv, per_region_reference_stats.csv,
visual_metric_stats.csv, primary_cell_claims.csv, per_region_cell_claims.csv,
audit_summary.json. Run dir: d_ablation_results.csv, pe_ablation_results.csv,
single_vs_proposed_current_protocol_south_korea.csv. Optional: the tau
re-sweep CSV (scripts/tau_resweep_d02.py).
Outputs: <stats dir>/tables/*.tex
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import os

from _paths import RUN, RUNS_DIR, STATS_DIR as HERE, TABLES_DIR as OUT

OUT.mkdir(parents=True, exist_ok=True)
TAU_CSV = Path(os.environ.get("TAU_RESWEEP_OUT", RUNS_DIR / "tau_resweep_d02")) / "tau_resweep_d02_retrieval.csv"
if not TAU_CSV.exists():
    TAU_CSV = HERE / "tau_resweep_d02_retrieval.csv"

TOP_K = (1, 3, 5, 10, 50, 100)
SHOW_K_VISUAL = (1, 5, 10, 100)
BB = ("CLIP", "ConvNeXt2", "DINOv2")

main = pd.read_csv(HERE / "primary_main_stats.csv")
perreg = pd.read_csv(HERE / "per_region_reference_stats.csv")
vis = pd.read_csv(HERE / "visual_metric_stats.csv")
cells = pd.read_csv(HERE / "primary_cell_claims.csv")
prcells = pd.read_csv(HERE / "per_region_cell_claims.csv")
summary = json.loads((HERE / "audit_summary.json").read_text())

REGION_LABEL = {
    "south_korea": r"\texttt{south\_korea}",
    "south_korea_wide": r"\texttt{south\_korea\_wide}",
    "low_left": r"\texttt{low\_left}",
    "total_patches": r"\texttt{total\_patches}",
}
SHORT = {"Proposed": "RaP-Weather", "RaP-Weather": "RaP-Weather", "Pixel-NN": "Pixel-NN"}
for bb in BB:
    SHORT[f"{bb} (Group1, off-the-shelf)"] = f"{bb} (G1)"
    SHORT[f"{bb} (Group2, whole-trained + crop)"] = f"{bb} (G2)"
    SHORT[f"{bb} (Group2, whole-image fine-tuned)"] = f"{bb} (G2)"
    SHORT[f"{bb} (Group3, whole-trained + crop)"] = f"{bb} (G2)"
    SHORT[f"{bb} (Group2, predictive fine-tuned)"] = f"{bb} (G2)"
    SHORT[f"{bb} (per-region)"] = f"{bb} (per-region)"


def f4(x): return f"{x:.4f}"
def f3(x): return f"{x:.3f}"


def fp(p):
    """Holm p for tables: 3 decimals, or scientific below 1e-3."""
    if pd.isna(p): return "--"
    if p >= 1.0: return "1.000"
    if p >= 1e-3: return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


MARK_BOX = r"\makebox[0.55em][l]{%s}"   # fixed-width slot so numbers stay aligned with or without a marker

def marker(row, rap_mean, lower_is_better=True):
    """Fixed-width marker slot: * = Holm p<0.05 and RaP better; † = Holm p<0.05 and baseline better;
    empty slot otherwise. Every numeric cell (RaP-Weather rows included) gets the slot so that the
    digits line up within a column regardless of the marker."""
    if pd.isna(row.get("Holm p")) or row["Holm p"] >= 0.05:
        return MARK_BOX % ""
    base_better = (row["Mean"] < rap_mean) if lower_is_better else (row["Mean"] > rap_mean)
    return MARK_BOX % (r"$^{\dagger}$" if base_better else r"$^{*}$")


def write(name, text):
    (OUT / name).write_text(text.rstrip() + "\n")
    print("wrote", name)


# ---------------------------------------------------------------- static tables
write("tab_impl_details.tex", r"""
\begin{table}[t]
\caption{Implementation details of RaP-Weather and the learned baselines.}
\label{tab:impl_details}
\centering
{\tablebodyfont\fontsize{7.5}{9}\selectfont
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l@{\hspace{5pt}}>{\raggedright\arraybackslash}p{\dimexpr\linewidth-90pt-5pt\relax}@{}}
\toprule
Component & Setting \\
\midrule
Backbone & ViT-B/16 (\texttt{vit\_base\_patch16\_224}), ImageNet-pretrained \\
LoRA & $r{=}16$, $\alpha_{\mathrm{LoRA}}{=}32$, dropout 0.1, targeting attention \texttt{qkv} \\
MLP bottleneck & $768 \to 512 \to 256 \to 128 \to 8$ (GELU) \\
Input normalization & fixed $[0,255] \to [0,1]$ (no ImageNet mean/std) \\
Sequence length / horizon & 6 frames (15\,h span) / $+1$ step (3\,h) \\
Temporal Transformer & 2 layers, 4 heads, FFN 128, pre-norm, dropout 0.2 \\
Batch size / epochs & 64 / 10 \\
Learning rates & backbone \& bottleneck $5{\times}10^{-6}$, temporal \& head $3{\times}10^{-4}$ \\
Optimizer / scheduler & AdamW / none \\
Precision & mixed precision (AMP), gradient clipping 1.0 \\
Loss weights & $\alpha{=}0.1$, $\tau{=}0.07$ \\
Hardware & RTX 4090, CUDA 11.8, PyTorch 2.6.0, Python 3.12.2 \\
Training / inference seed & 42 \\
Model-selection seed & 123 (200 queries, excluded from primary evaluation) \\
\midrule
Baseline checkpoints & CLIP-B/32 {\fontsize{6.5}{8}\selectfont\texttt{vit\_base\_patch32\_clip\_224.openai}} \\
 & ConvNeXt V2-T {\fontsize{6.5}{8}\selectfont\texttt{convnextv2\_tiny.fcmae\_ft\_in22k\_in1k}} \\
 & DINOv2-S/14 {\fontsize{6.5}{8}\selectfont\texttt{vit\_small\_patch14\_dinov2.lvd142m}} \\
Baseline feature & \texttt{timm} pooled output with \texttt{num\_classes=0}; 768/768/384 dimensions \\
Group 1 tuning & none; all backbone parameters frozen \\
Group 2 tuning & final 20 backbone parameter tensors trainable; no LoRA or projection \\
Group 2 temporal module & 2 layers, 4 heads, FFN 128, pre-norm, dropout 0.1 \\
Group 2 optimization & AdamW; $5{\times}10^{-6}$ backbone, $3{\times}10^{-4}$ temporal/head; no scheduler \\
Baseline normalization & fixed $[0,255] \to [0,1]$; no model-specific mean/std \\
\bottomrule
\end{tabular}
}
\end{table}
""")

write("tab_lora.tex", r"""
\begin{table}[t]
\caption{Full LoRA rank and scaling grid on the 200 model-selection queries. Lower validation retrieval MSE is better.}
\label{tab:lora_grid}
\centering
{\tablebodyfont\fontsize{7.5}{9}\selectfont
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}cccc@{\hspace{1em}}cccc@{}}
\toprule
$r$ & $\alpha_{\mathrm{LoRA}}$ & ratio & Val.\ MSE & $r$ & $\alpha_{\mathrm{LoRA}}$ & ratio & Val.\ MSE \\
\midrule
16 & 32 & 2.0 & \textbf{1.0551} & 32 & 128 & 4.0 & 1.2383 \\
8 & 8 & 1.0 & 1.0766 & 32 & 64 & 2.0 & 1.2833 \\
4 & 16 & 4.0 & 1.0956 & 16 & 8 & 0.5 & 1.3231 \\
16 & 16 & 1.0 & 1.1014 & 8 & 16 & 2.0 & 1.3317 \\
8 & 32 & 4.0 & 1.1186 & 4 & 2 & 0.5 & 1.4444 \\
32 & 32 & 1.0 & 1.1740 & 32 & 16 & 0.5 & 1.4523 \\
16 & 64 & 4.0 & 1.1768 & 4 & 4 & 1.0 & 1.4765 \\
8 & 4 & 0.5 & 1.1924 & 4 & 8 & 2.0 & 1.8521 \\
\bottomrule
\end{tabular*}}
\end{table}
""")

# ---------------------------------------------------------------- alpha-tau (+ tau re-check rows)
alpha_tau_rows = [
    ("0.10", "0.07", r"\textbf{1.0551}", "0.05", "0.10", "1.1802"),
    ("0.05", "0.20", "1.0559", "0.10", "0.20", "1.1846"),
    ("0.10", "0.10", "1.0675", "0.50", "0.10", "1.1862"),
    ("0.50", "0.07", "1.0755", "0.05", "0.05", "1.1941"),
    ("0.50", "0.05", "1.1494", "1.00", "0.05", "1.1976"),
    ("0.10", "0.05", "1.1503", "1.00", "0.07", "1.2398"),
    ("0.05", "0.07", "1.1726", "1.00", "0.10", "1.2504"),
    ("1.00", "0.20", "1.1731", "0.50", "0.20", "1.2634"),
]
tau_block = ""
tau_txt_iv = ""
tau_txt_app = ""
if TAU_CSV.exists():
    t = pd.read_csv(TAU_CSV).sort_values("tau")
    best = t.sort_values("val_retrieval_mse").iloc[0]
    rows = " \\\\\n".join(
        f"0.10 & {r.tau:.2f} & " + (r"\textbf{" + f4(r.val_retrieval_mse) + "}" if r.tau == best.tau else f4(r.val_retrieval_mse))
        for r in t.itertuples())
    tau_block = (r"\midrule" "\n"
                 r"\multicolumn{6}{@{}l}{\emph{$\tau$ re-check at temporal dropout 0.2 ($\alpha{=}0.1$, same 200 queries)}} \\" "\n"
                 r"\multicolumn{3}{@{}c}{" + rows.replace(" \\\\\n", r"} \\ \multicolumn{3}{@{}c}{").replace("0.10 & ", "$\\alpha$=0.10, $\\tau$=") + r"} \\" "\n")
    # simpler: one row per tau across the 6 columns
    tau_block = "\\midrule\n" + r"\multicolumn{6}{@{}l}{\emph{$\tau$ re-check at temporal dropout 0.2 ($\alpha{=}0.1$, same 200 queries)}} \\" + "\n"
    tl = list(t.itertuples())
    for i in range(0, len(tl), 2):
        a = tl[i]; b = tl[i + 1] if i + 1 < len(tl) else None
        def cell(r):
            v = f4(r.val_retrieval_mse)
            return r"\textbf{" + v + "}" if r.tau == best.tau else v
        left = f"0.10 & {a.tau:.2f} & {cell(a)}"
        right = f"0.10 & {b.tau:.2f} & {cell(b)}" if b is not None else " & & "
        tau_block += f"{left} & {right} \\\\\n"
    best_tau = float(best.tau)
    t_sorted = t.sort_values("val_retrieval_mse")
    score_items = [f"{f4(r.val_retrieval_mse)} ($\\tau{{=}}{r.tau:g}$)" for r in t_sorted.itertuples()]
    scores = ", ".join(score_items[:-1]) + ", and " + score_items[-1] if len(score_items) > 2 else " and ".join(score_items)
    sel = float(t[t["tau"] == 0.07]["val_retrieval_mse"].iloc[0])
    gap = sel - float(best.val_retrieval_mse)
    REPORTED_TAU = 0.07
    if abs(best_tau - REPORTED_TAU) < 1e-9:
        tau_txt_iv = (r"The balancing coefficient $\alpha$ and temperature $\tau$ are fixed hyperparameters rather than learnable weights. We selected them through a $4 \times 4$ grid search over $\alpha \in \{0.05, 0.1, 0.5, 1.0\}$ and $\tau \in \{0.05, 0.07, 0.1, 0.2\}$. Each combination was trained \revb{with seed 42} for 3 epochs and ranked by its mean Top-1/5/10 weather-observation MSE on 200 \texttt{south\_korea} model-selection queries from the chronological future partition (sampled with seed 123). The selected queries are excluded from the 4,943-query confirmatory comparison in Section~\ref{sec:eval_protocol}. The grid yielded $\alpha{=}0.1$ and $\tau{=}0.07$ (score 1.0551). This grid was run with a temporal dropout of 0.1\revb{. After} the temporal dropout was set to 0.2, we re-checked $\tau$ on the same 200 queries, which again selected $\tau{=}0.07$. Both sweeps are reported in Appendix~\ref{app:sweeps}.")
        tau_txt_app = (r"The lower block of Table~\ref{tab:alpha_tau_sweep} re-checks $\tau$ at the final temporal dropout of 0.2 with $\alpha{=}0.1$ fixed, on the same 200 queries and the same 3-epoch protocol\revb{. The} scores are " + scores + r". The selected value $\tau{=}0.07$ is unchanged.")
    else:
        # Reported model keeps tau=0.07; the re-check ranks another value first by a small margin.
        tau_txt_iv = (r"The balancing coefficient $\alpha$ and temperature $\tau$ are fixed hyperparameters rather than learnable weights. We selected them through a $4 \times 4$ grid search over $\alpha \in \{0.05, 0.1, 0.5, 1.0\}$ and $\tau \in \{0.05, 0.07, 0.1, 0.2\}$. Each combination was trained \revb{with seed 42} for 3 epochs and ranked by its mean Top-1/5/10 weather-observation MSE on 200 \texttt{south\_korea} model-selection queries from the chronological future partition (sampled with seed 123). The selected queries are excluded from the 4,943-query confirmatory comparison in Section~\ref{sec:eval_protocol}. The grid yielded $\alpha{=}0.1$ and $\tau{=}0.07$ (score 1.0551), with 1.0559 ($\alpha{=}0.05, \tau{=}0.2$) and 1.0675 ($\alpha{=}0.1, \tau{=}0.1$) next. "
                      + f"This grid was run with a temporal dropout of 0.1. \\revb{{After the temporal dropout was set to 0.2, we re-checked $\\tau$ on the same 200 queries and kept $\\tau{{=}}0.07$, since the best alternative differed by only {gap:.4f} (Appendix~\\ref{{app:sweeps}}).}}")
        tau_txt_app = (r"The lower block of Table~\ref{tab:alpha_tau_sweep} re-checks $\tau$ at the final temporal dropout of 0.2 with $\alpha{=}0.1$ fixed, on the same 200 queries and the same 3-epoch protocol. The scores are " + scores + "."
                       + f" The best value, $\\tau{{=}}{best_tau:g}$, is ahead of the reported $\\tau{{=}}0.07$ by only {gap:.4f}\\revb{{. For}} reference, the two best settings of the original grid differ by 0.0008. All results in this paper use $\\tau{{=}}0.07$.")
else:
    tau_txt_iv = "\\textbf{[TAU RE-SWEEP PENDING: rerun make\\_tables.py after the sweep finishes]}"
    tau_txt_app = tau_txt_iv

body = " \\\\\n".join(" & ".join(r) for r in alpha_tau_rows) + " \\\\\n"
write("tab_alpha_tau.tex", r"""
\begin{table}[t]
\caption{Full joint grid for $\alpha$ and $\tau$ on the 200 model-selection queries (temporal dropout 0.1), followed by the $\tau$ re-check at the final temporal dropout of 0.2. Lower validation retrieval MSE is better.}
\label{tab:alpha_tau_sweep}
\centering
{\tablebodyfont\fontsize{7.5}{9}\selectfont
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}ccc@{\hspace{1.5em}}ccc@{}}
\toprule
$\alpha$ & $\tau$ & Val.\ MSE & $\alpha$ & $\tau$ & Val.\ MSE \\
\midrule
""" + body + tau_block + r"""\bottomrule
\end{tabular*}}
\end{table}
""")
write("txt_alpha_tau.tex", tau_txt_iv)
write("txt_appendix_tau.tex", tau_txt_app)

# ---------------------------------------------------------------- Top-1 MSE (south_korea, total_patches)
def mse_top1_block(scope, title):
    d = main[(main["Scope"] == scope) & (main["Top-k"] == 1)]
    rap = d[d["Model"] == "Proposed"].iloc[0]
    best = d["Mean"].min()
    lines = [r"\multicolumn{4}{@{}l}{\emph{" + title + r"}} \\"]
    for r in d.itertuples():
        name = SHORT[r.Model]
        mean = f4(r.Mean)
        if abs(r.Mean - best) < 5e-5: mean = r"\textbf{" + mean + "}"
        if r.Model == "Proposed": name = r"\textbf{RaP-Weather}"
        ci = f"[{f4(r._5)}, {f4(r._6)}]"  # superseded by the redefinition below
        lines.append(f"{name} & {mean} & [{f4(getattr(r, '_5'))}, {f4(getattr(r, '_6'))}] & {fp(getattr(r, '_9', np.nan) if False else r._asdict().get('Holm p', np.nan))} \\\\")
    return "\n".join(lines)

# itertuples renames columns with spaces; use explicit iteration instead
def mse_top1_block(scope, title):
    d = main[(main["Scope"] == scope) & (main["Top-k"] == 1)]
    best = d["Mean"].min()
    lines = [r"\multicolumn{4}{@{}l}{\emph{" + title + r"}} \\"]
    for _, r in d.iterrows():
        name = r"\textbf{RaP-Weather}" if r["Model"] == "Proposed" else SHORT[r["Model"]]
        mean = f4(r["Mean"])
        if abs(r["Mean"] - best) < 5e-5: mean = r"\textbf{" + mean + "}"
        lines.append(f"{name} & {mean} & [{f4(r['95% CI Low'])}, {f4(r['95% CI High'])}] & {fp(r.get('Holm p', np.nan))} \\\\")
    return "\n".join(lines)

write("tab_mse_top1.tex", r"""
\begin{table}[!t]
\caption{Top-1 weather-observation MSE with bootstrap 95\% CIs over queries and Holm-adjusted paired Wilcoxon $p$-values vs.\ RaP-Weather ($N{=}4{,}943$ confirmatory queries). Bold marks the best mean per query type. On the full-image query the crop is the identity, so Group~2 coincides with the whole-image fine-tuned encoder.}
\label{tab:mse_top1}
\centering
{\tablebodyfont\fontsize{7.5}{9}\selectfont
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccc@{}}
\toprule
Model & Top-1 MSE & 95\% CI & Holm $p$ \\
\midrule
""" + mse_top1_block("south_korea", r"Regional query (\texttt{south\_korea})") + "\n\\midrule\n"
      + mse_top1_block("total_patches", r"Full-image query (\texttt{total\_patches})") + r"""
\bottomrule
\end{tabular*}
}
\end{table}
""")

# ---------------------------------------------------------------- full MSE with markers
def full_block(scope):
    d = main[main["Scope"] == scope]
    lines = [r"\multicolumn{7}{@{}l}{\emph{" + REGION_LABEL[scope] + r"}} \\"]
    models = list(dict.fromkeys(d["Model"]))
    for m in models:
        cellsx = []
        for k in TOP_K:
            r = d[(d["Model"] == m) & (d["Top-k"] == k)].iloc[0]
            rap = d[(d["Model"] == "Proposed") & (d["Top-k"] == k)].iloc[0]["Mean"]
            best = d[d["Top-k"] == k]["Mean"].min()
            v = f4(r["Mean"])
            if abs(r["Mean"] - best) < 5e-5: v = r"\textbf{" + v + "}"
            v += marker(r, rap) if m != "Proposed" else MARK_BOX % ""
            cellsx.append(v)
        name = r"\textbf{RaP-Weather}" if m == "Proposed" else SHORT[m]
        lines.append(name + " & " + " & ".join(cellsx) + r" \\")
    return "\n".join(lines)

# Table 4 as a single-column table: 7.5 pt, 1.5 pt column padding, short model labels
# ("CLIP G1" instead of "CLIP (G1)") so that all six depths fit in \linewidth.
def full_block_1col(scope):
    return full_block(scope).replace(" (G1)", " G1").replace(" (G2)", " G2")

write("tab_full_mse.tex", r"""
\begin{table}[!t]
\caption{Mean weather-observation MSE for all query configurations ($N{=}4{,}943$ confirmatory queries). All methods shown serve arbitrary regions without per-region training. On \texttt{total\_patches}, Group~2 uses the whole-image fine-tuned encoders with the identity crop. Bold marks the best mean per column. \revb{An asterisk ($^{*}$) marks Holm-adjusted paired Wilcoxon $p<0.05$ in favor of RaP-Weather, a dagger ($^{\dagger}$) marks $p<0.05$ in favor of the baseline, and unmarked cells are not significant.}}
\label{tab:full_mse}
\centering
{\tablebodyfont\fontsize{7.5}{9}\selectfont
\setlength{\tabcolsep}{1.5pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
Model & Top-1 & Top-3 & Top-5 & Top-10 & Top-50 & Top-100 \\
\midrule
""" + "\n\\midrule\n".join(full_block_1col(s) for s in REGION_LABEL) + r"""
\bottomrule
\end{tabular*}
}
\end{table}
""")

# ---------------------------------------------------------------- per-region encoders (Appendix B)
def g2_block(scope):
    d = perreg[perreg["Scope"] == scope]
    lines = [r"\multicolumn{9}{@{}l}{\emph{" + REGION_LABEL[scope] + r"}} \\"]
    for m in list(dict.fromkeys(d["Model"])):
        cellsx = []
        for k in TOP_K:
            r = d[(d["Model"] == m) & (d["Top-k"] == k)].iloc[0]
            rap = d[(d["Model"] == "Proposed") & (d["Top-k"] == k)].iloc[0]["Mean"]
            best = d[d["Top-k"] == k]["Mean"].min()
            v = f4(r["Mean"])
            if abs(r["Mean"] - best) < 5e-5: v = r"\textbf{" + v + "}"
            v += marker(r, rap) if m != "Proposed" else MARK_BOX % ""
            cellsx.append(v)
        r1 = d[(d["Model"] == m) & (d["Top-k"] == 1)].iloc[0]
        ci = f"[{f4(r1['95% CI Low'])}, {f4(r1['95% CI High'])}]"
        p = fp(r1.get("Holm p", np.nan)) if m != "Proposed" else "--"
        name = "RaP-Weather" if m == "Proposed" else SHORT[m]
        lines.append(name + " & " + " & ".join(cellsx) + f" & {ci} & {p} \\\\")
    return "\n".join(lines)

write("tab_g2_full.tex", r"""
\begin{table*}[t]
\caption{Mean weather-observation MSE of the per-region fine-tuned encoders ($N{=}4{,}943$ confirmatory queries), with bootstrap 95\% CIs over queries and Holm-adjusted paired Wilcoxon $p$-values vs.\ RaP-Weather at Top-1. Bold marks the best mean per column\revb{. Markers are} as in Table~\ref{tab:full_mse}.}
\label{tab:g2_full}
\centering
{\tablebodyfont\fontsize{7.5}{9}\selectfont
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc@{}}
\toprule
Model & Top-1 & Top-3 & Top-5 & Top-10 & Top-50 & Top-100 & Top-1 95\% CI & Holm $p$ \\
\midrule
""" + "\n\\midrule\n".join(g2_block(s) for s in REGION_LABEL) + r"""
\bottomrule
\end{tabular*}
}
\end{table*}
""")

# ---------------------------------------------------------------- visual metrics (G1 + G2), @1 with CI, markers
def vis_rows(qtype):
    d = vis[vis["Query Type"] == qtype]
    models = list(dict.fromkeys(d["Model"]))
    out = []
    for m in models:
        cellsx = []
        for metric in ("LPIPS", "PSNR", "SSIM"):
            lower = metric == "LPIPS"
            for k in SHOW_K_VISUAL:
                r = d[(d["Model"] == m) & (d["Metric"] == metric) & (d["Top-k"] == k)].iloc[0]
                col = d[(d["Metric"] == metric) & (d["Top-k"] == k)]
                best = col["Mean"].min() if lower else col["Mean"].max()
                rap = col[col["Model"] == "RaP-Weather"].iloc[0]["Mean"]
                v = f4(r["Mean"]) if metric != "PSNR" else f3(r["Mean"])
                if abs(r["Mean"] - best) < (5e-5 if metric != "PSNR" else 5e-4): v = r"\textbf{" + v + "}"
                v += marker(r, rap, lower_is_better=lower) if m != "RaP-Weather" else MARK_BOX % ""
                if k == 1:
                    lo, hi = (f4(r["95% CI Low"]), f4(r["95% CI High"])) if metric != "PSNR" else (f3(r["95% CI Low"]), f3(r["95% CI High"]))
                    v = r"\shortstack{" + v + r"\\[-1pt]{\tiny[" + lo + r",\," + hi + "]}}"
                cellsx.append(v)
        name = r"\textbf{RaP-Weather}" if m == "RaP-Weather" else SHORT[m]
        out.append("& " + name + " & " + " & ".join(cellsx) + r" \\")
    return out

full_rows = vis_rows("Full Image"); reg_rows = vis_rows("Region")
write("tab_visual.tex", r"""
\begin{table*}[t]
\centering
\caption{Average visual-similarity scores for Top-$k$ retrieved images by query type over the 4,943 confirmatory queries. \revb{G1 denotes the off-the-shelf encoders and G2 denotes the whole-image-trained encoders with crops resized at query time (identity crop for full-image queries).} Top-1 cells show the bootstrap 95\% CI over queries. Bold marks the best value per column. \revb{An asterisk ($^{*}$) marks Holm-adjusted paired Wilcoxon $p<0.05$ in favor of RaP-Weather, a dagger ($^{\dagger}$) marks $p<0.05$ in favor of the baseline, and unmarked cells are not significant.}}
\label{tab:combined_quant_metrics}
{\tablebodyfont\fontsize{7}{8.5}\selectfont
\setlength{\tabcolsep}{2.5pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll cccc cccc cccc@{}}
\toprule
          &           & \multicolumn{4}{c}{Avg LPIPS $\downarrow$} & \multicolumn{4}{c}{Avg PSNR $\uparrow$} & \multicolumn{4}{c}{Avg SSIM $\uparrow$} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-14}
Query Type & Model     & @1      & @5     & @10    & @100   & @1      & @5     & @10    & @100   & @1      & @5     & @10    & @100   \\
\midrule
\multirow{""" + str(len(full_rows)) + r"""}{*}{Full Image}
""" + "\n".join(full_rows) + r"""
\midrule
\multirow{""" + str(len(reg_rows)) + r"""}{*}{Region}
""" + "\n".join(reg_rows) + r"""
\bottomrule
\end{tabular*}
}
\end{table*}
""")

# ---------------------------------------------------------------- design table (5,143, descriptive)
d_ab = pd.read_csv(RUN / "d_ablation_results.csv")
pe_ab = pd.read_csv(RUN / "pe_ablation_results.csv")
sv = pd.read_csv(RUN / "single_vs_proposed_current_protocol_south_korea.csv")

def ci_cell(mean, ci_str_or_pair, bold=False):
    if isinstance(ci_str_or_pair, str):
        lo, hi = [float(x) for x in re.findall(r"[\d.]+", ci_str_or_pair)]
    else:
        lo, hi = ci_str_or_pair
    m = f4(mean); c = f"[{f4(lo)},\\,{f4(hi)}]"
    if bold: m = r"\textbf{" + m + "}"; c = r"\textbf{" + c + "}"
    return r"\shortstack{" + m + r"\\[-1pt]{\fontsize{6}{7}\selectfont " + c + "}}"  # design table has wide columns; 6 pt CI stays legible

design_lines = []
best_d = {k: d_ab[f"Top-{k} MSE"].min() for k in TOP_K}
for _, r in d_ab.iterrows():
    cellsx = [ci_cell(r[f"Top-{k} MSE"], r[f"Top-{k} 95% CI"], bold=abs(r[f"Top-{k} MSE"] - best_d[k]) < 5e-5) for k in TOP_K]
    label = f"RaP-Weather, $D{{=}}{int(r['D'])}$" + (" (default)" if int(r["D"]) == 8 else "")
    design_lines.append(label + " & " + " & ".join(cellsx) + r" \\")
design_lines.append(r"\midrule")
pe_ret = pe_ab[pe_ab["Variant"].str.startswith("w/ ")].iloc[0]
design_lines.append("RaP-Weather, $D{=}8$, PE retained & " + " & ".join(ci_cell(pe_ret[f"Top-{k} MSE"], pe_ret[f"Top-{k} 95% CI"]) for k in TOP_K) + r" \\")
design_lines.append("RaP-Single Image ($D{=}8$) & " + " & ".join(
    ci_cell(sv[sv["Top-k"] == k].iloc[0]["RaP-Single MSE"], (sv[sv["Top-k"] == k].iloc[0]["Single CI Low"], sv[sv["Top-k"] == k].iloc[0]["Single CI High"])) for k in TOP_K) + r" \\")

write("tab_design.tex", r"""
\begin{table*}[t]
\caption{Design analyses on all 5,143 \texttt{south\_korea} queries\revb{. These analyses are descriptive and not part of the confirmatory family.} Mean observation MSE with bootstrap 95\% CI over queries. The $D{=}8$ row is the proposed model of Table~\ref{tab:mse_top1}\revb{. ``PE} retained'' keeps the ViT positional encoding before the bottleneck\revb{. RaP-Single} Image replaces the predictive objective with single-image InfoNCE (Section~\ref{sec:Impact_of_Spatial_temporal}). Bold marks the lowest mean among the $D$ variants.}
\label{tab:design_exact}
\centering
{\tablebodyfont\fontsize{7.5}{9}\selectfont
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
Variant & Top-1 & Top-3 & Top-5 & Top-10 & Top-50 & Top-100 \\
\midrule
""" + "\n".join(design_lines) + r"""
\bottomrule
\end{tabular*}
}
\end{table*}
""")

# ---------------------------------------------------------------- claim sentences (derived from CSVs)
n_lowest = int(summary["cells_rap_lowest_mean"]); n_beats = int(summary["cells_rap_beats_all"]); n_loss = int(summary["cells_rap_significant_loss"])
assert n_loss == 0, "RaP has a significant loss in the main family: rewrite the scoped claim manually"

def natural_list(items):
    """['a'] -> 'a'; ['a','b'] -> 'a and b'; ['a','b','c'] -> 'a, b, and c'."""
    items = list(items)
    if len(items) <= 1: return "".join(items)
    if len(items) == 2: return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def ks_label(ks):
    return "Top-" + "/".join(str(int(k)) for k in sorted(set(ks)))


def cell_list(mask):
    """Human-readable list of (scope, k) cells."""
    by = {}
    for _, c in cells[mask].iterrows():
        by.setdefault(c["Scope"], []).append(int(c["Top-k"]))
    parts = []
    for s, ks in by.items():
        if ks == list(TOP_K): parts.append(f"every depth on {REGION_LABEL[s]}")
        else: parts.append("Top-" + "/".join(map(str, ks)) + f" on {REGION_LABEL[s]}")
    return ", and ".join(parts) if len(parts) <= 2 else ", ".join(parts[:-1]) + ", and " + parts[-1]

beats_desc = cell_list(cells["RaP beats all"])
not_lowest = cells[~cells["RaP lowest mean"]]
sk1 = main[(main["Scope"] == "south_korea") & (main["Top-k"] == 1)].set_index("Model")
tp1 = main[(main["Scope"] == "total_patches") & (main["Top-k"] == 1)].set_index("Model")
rap_sk = sk1.loc["Proposed"]; rap_tp = tp1.loc["Proposed"]
best_g1_sk = sk1.loc[[f"{b} (Group1, off-the-shelf)" for b in BB]]["Mean"].idxmin()
best_g2_sk = sk1.loc[[f"{b} (Group2, whole-trained + crop)" for b in BB]]["Mean"].idxmin()
dino_tp = tp1.loc["DINOv2 (Group2, whole-image fine-tuned)"]

abstract_claim = (f"no baseline is significantly better than RaP-Weather at any query configuration or ranking depth, and RaP-Weather attains the lowest meteorological-observation MSE in {n_lowest} of 24 region--depth combinations \\revb{{on 4,943 held-out queries}}, significantly so against every comparator in {n_beats}.").replace("meteorological-observation MSE", "\\revb{mean squared error against meteorological observations}")
write("txt_abstract_claim.tex", abstract_claim)

conclusion_claim = (f"Among methods that support arbitrary regions without per-region training, no baseline is significantly better than RaP-Weather at any ranking depth. On the 4,943 queries not used for model selection, RaP-Weather has the lowest meteorological-observation MSE in {n_lowest} of 24 region--depth combinations and is significantly better than every comparator in {n_beats} after Holm correction\\revb{{. Its}} advantage therefore lies in deeper ranking and in reuse across regions rather than in single-best-match accuracy.")
write("txt_conclusion_claim.tex", conclusion_claim)

nonsig_sk = [SHORT[m] for m in sk1.index if m != "Proposed" and sk1.loc[m, "Holm p"] >= 0.05]
nonsig_tp = [(SHORT[m], sk) for m in tp1.index if m != "Proposed" and tp1.loc[m, "Holm p"] >= 0.05 for sk in [None]]
def nonsig_desc(scope):
    """Compact: 'Top-1 (three encoders) and Top-3/5 (DINOv2 G2)'."""
    d = main[(main["Scope"] == scope) & (main["Model"] != "Proposed") & (main["Holm p"] >= 0.05)]
    if d.empty: return ""
    by_k = {}
    for _, r in d.iterrows():
        by_k.setdefault(int(r["Top-k"]), []).append(SHORT[r["Model"]])
    # merge depths that share the same model set: 'DINOv2 (G2) at Top-3/5'
    by_models = {}
    for k in sorted(by_k):
        by_models.setdefault(tuple(sorted(by_k[k])), []).append(k)
    parts = []
    for names, ks in by_models.items():
        who = names[0] if len(names) == 1 else f"{NUM_WORDS.get(len(names), len(names))} of the compared encoders"
        parts.append(f"{who} at {ks_label(ks)}")
    return natural_list(parts)
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
_region_parts = [f"{nonsig_desc(s_)} on {REGION_LABEL[s_]}" for s_ in REGION_LABEL if nonsig_desc(s_)]
_roman = ("i", "ii", "iii", "iv")
if len(_region_parts) > 1:
    _num = [f"\\revb{{({_roman[i]})}} {part}" for i, part in enumerate(_region_parts)]
    nonsig_by_region = "\\revb{,} ".join(_num[:-1]) + "\\revb{, and} " + _num[-1]
else:
    nonsig_by_region = _region_parts[0]
n_nonsig_cells = int((~cells["RaP beats all"]).sum())
assert n_nonsig_cells == 24 - n_beats

mse_results = (
    f"The comparison contains RaP-Weather, the off-the-shelf Group~1 encoders, the whole-image-trained Group~2 encoders, and Pixel-NN. "
    f"Among these methods, which support arbitrary regions without per-region training, no baseline is significantly better than RaP-Weather in any of the 24 region--depth combinations after Holm correction. "
    f"RaP-Weather has the lowest mean in {n_lowest} of the 24 combinations and is significantly better than every comparator in {n_beats}\\revb{{. These are}} {beats_desc}. "
    f"On {REGION_LABEL['south_korea']}, Top-1 MSE is {f4(rap_sk['Mean'])} [{f4(rap_sk['95% CI Low'])}, {f4(rap_sk['95% CI High'])}] versus {f4(sk1.loc[best_g1_sk,'Mean'])} for the best Group~1 model ({SHORT[best_g1_sk].split(' (')[0]}, Holm $p{{=}}{sk1.loc[best_g1_sk,'Holm p']:.3f}$), {f4(sk1.loc[best_g2_sk,'Mean'])} for the best Group~2 model ({SHORT[best_g2_sk].split(' (')[0]}, $p{{=}}{sk1.loc[best_g2_sk,'Holm p']:.2f}$), and {f4(sk1.loc['Pixel-NN','Mean'])} for Pixel-NN. "
    f"The {n_nonsig_cells} combinations without a significant RaP-Weather advantage over every comparator are all at shallow depth, namely {nonsig_by_region}. "
    f"On {REGION_LABEL['total_patches']}, where Group~2 coincides with the whole-image fine-tuned encoder, DINOv2 attains a marginally lower Top-1 mean ({f4(dino_tp['Mean'])} versus {f4(rap_tp['Mean'])}, $p{{=}}{dino_tp['Holm p']:.3f}$)\\revb{{. From}} Top-50 onward RaP-Weather is significantly better than every comparator in every configuration."
)
write("txt_mse_results.tex", mse_results)

# visual results sentence
def vrow(qt, metric, model, k=1):
    return vis[(vis["Query Type"] == qt) & (vis["Metric"] == metric) & (vis["Model"] == model) & (vis["Top-k"] == k)].iloc[0]
full_all_sig = all(vrow("Full Image", m, mo)["Holm p"] < 0.05 and not vrow("Full Image", m, mo)["Baseline better"]
                   for m in ("LPIPS", "PSNR", "SSIM") for mo in vis[(vis["Query Type"] == "Full Image")]["Model"].unique() if mo != "RaP-Weather")
reg_lpips_sig = [SHORT[mo] for mo in vis[(vis["Query Type"] == "Region")]["Model"].unique() if mo != "RaP-Weather"
                 and vrow("Region", "LPIPS", mo)["Holm p"] < 0.05 and vrow("Region", "LPIPS", mo)["Baseline better"]]
reg_psnr_better = [(SHORT[mo], vrow("Region", "PSNR", mo)) for mo in vis[(vis["Query Type"] == "Region")]["Model"].unique() if mo != "RaP-Weather"
                   and vrow("Region", "PSNR", mo)["Holm p"] < 0.05 and vrow("Region", "PSNR", mo)["Baseline better"]]
reg_ssim_better = [(SHORT[mo], vrow("Region", "SSIM", mo)) for mo in vis[(vis["Query Type"] == "Region")]["Model"].unique() if mo != "RaP-Weather"
                   and vrow("Region", "SSIM", mo)["Holm p"] < 0.05 and vrow("Region", "SSIM", mo)["Baseline better"]]
rap_reg_psnr = vrow("Region", "PSNR", "RaP-Weather")["Mean"]; rap_reg_ssim = vrow("Region", "SSIM", "RaP-Weather")["Mean"]
vis_txt = ("The quantitative results in Table~\\ref{tab:combined_quant_metrics} reveal different behavior for the two query types. For full-image queries, RaP-Weather achieves the best Top-1 score on LPIPS, PSNR, and SSIM"
           + (", and every Top-1 difference is significant after Holm correction. " if full_all_sig else ". ")
           + "For cropped region queries, " + (f"all {len(reg_lpips_sig)} compared encoders yield significantly lower (better) LPIPS than RaP-Weather" if len(reg_lpips_sig) == 6 else ", ".join(reg_lpips_sig) + " yield significantly lower (better) LPIPS")
           + (". On PSNR, " + " and ".join(f"{n} is significantly higher ({f3(r['Mean'])} versus {f3(rap_reg_psnr)}, $p{{=}}{fp(r['Holm p'])}$)".replace("$p{=}$", "$p{=}") for n, r in reg_psnr_better) if reg_psnr_better else ". On PSNR, no baseline is significantly higher than RaP-Weather")
           + ("\\revb{. On} SSIM, " + " and ".join(f"{n} is \\revb{{also}} significantly higher ({f4(r['Mean'])} versus {f4(rap_reg_ssim)}, $p{{=}}{fp(r['Holm p'])}$)" for n, r in reg_ssim_better) + "." if reg_ssim_better else "\\revb{. On} SSIM, no baseline is significantly higher than RaP-Weather.")
           + " \\revb{However, these} regional perceptual advantages of the baselines do not translate into meteorological similarity, as Section~\\ref{sec:Quantitative_Validation} shows.")
vis_txt = vis_txt.replace("$p{=}$", "$p{=}").replace("$$", "$")
write("txt_visual_results.tex", vis_txt)

# discussion divergence sentence
disc = (f"On regional queries, every compared encoder has significantly lower LPIPS than RaP-Weather, and {' and '.join(n for n,_ in reg_psnr_better) if reg_psnr_better else 'no encoder'} also has significantly higher PSNR (Section~\\ref{{sec:benchmark_eval}}). The ordering on the meteorological measure is different\\revb{{. On}} {REGION_LABEL['south_korea']}, RaP-Weather has significantly lower observation MSE than every Group~1 and Group~2 encoder from Top-3 onward, and no baseline is significantly better at any depth (Section~\\ref{{sec:Quantitative_Validation}}). Perceptual proximity and meteorological proximity are therefore not interchangeable on this data.")
write("txt_discussion_divergence.tex", disc)

# appendix B / C sentences
n_g1_sig = int(((main["Model"].str.contains("Group1") | (main["Model"] == "Pixel-NN")) & (main["Holm p"] < 0.05)).sum())
n_g1_tot = int((main["Model"].str.contains("Group1") | (main["Model"] == "Pixel-NN")).sum())
def grouped_exceptions(df, name_of):
    """Group non-significant rows as 'ConvNeXt2 and DINOv2 at Top-1 on low_left' per (scope, depth set)."""
    out = []
    for s_ in REGION_LABEL:
        d = df[df["Scope"] == s_]
        if d.empty: continue
        by_model = {}
        for _, r in d.iterrows():
            by_model.setdefault(name_of(r["Model"]), []).append(int(r["Top-k"]))
        # merge models that share the same depth set
        by_ks = {}
        for m, ks in by_model.items():
            by_ks.setdefault(tuple(sorted(ks)), []).append(m)
        for ks, ms in by_ks.items():
            out.append(f"{natural_list(ms)} at {ks_label(ks)} on {REGION_LABEL[s_]}")
    return natural_list(out)


# Counts only. Per-cell exceptions stay visible as markers in Table 4
# and as the eight-cell list in txt_mse_results.
_main_tested = main[main["Holm p"].notna()]
n_main_tot = int(len(_main_tested))
assert n_main_tot == int(summary["main_family_tests"]), (n_main_tot, summary["main_family_tests"])
n_main_sig = int(_main_tested["Holm significant"].astype(bool).sum())
assert n_main_sig == int(summary["main_family_holm_significant"]), (n_main_sig, summary["main_family_holm_significant"])
n_main_rest = n_main_tot - n_main_sig
k_max_nonsig = int(_main_tested[~_main_tested["Holm significant"].astype(bool)]["Top-k"].max())
n_base_win = int((_main_tested["Holm significant"].astype(bool) & _main_tested["Baseline mean lower"].astype(bool)).sum())
assert n_base_win == 0, "a baseline wins significantly; rewrite app_full"
app_full = (f"\\revb{{Of the {n_main_tot} comparisons in Table~\\ref{{tab:full_mse}}, {n_main_sig} favor RaP-Weather significantly. "
            f"The remaining {n_main_rest} are non-significant differences at Top-1 to Top-{k_max_nonsig}, and none is a significant advantage for a baseline.}} "
            f"Table~\\ref{{tab:mse_top1}} gives Top-1 confidence intervals and adjusted $p$-values.")
write("txt_appendix_full.tex", app_full)

pr_sig = prcells[prcells["Per-region significantly better"].notna()].rename(columns={"Per-region significantly better": "Model"})
pr_sig_desc = grouped_exceptions(pr_sig, lambda m: m.replace(" (per-region)", ""))
pr_lower = prcells[prcells["Per-region lower mean"].notna() & (prcells["Top-k"] <= 5)].rename(columns={"Per-region lower mean": "Model"})
pr_lower_desc = grouped_exceptions(pr_lower, lambda m: m.replace(" (per-region)", ""))
n_pr_lower = NUM_WORDS.get(len(pr_lower), str(len(pr_lower)))
n_pr_sig = NUM_WORDS.get(len(pr_sig), str(len(pr_sig)))
first_rap_best_k = {}
for s in REGION_LABEL:
    ks = prcells[(prcells["Scope"] == s) & prcells["RaP beats all per-region"]]["Top-k"]
    first_rap_best_k[s] = int(ks.min()) if len(ks) else None
clip_sk = perreg[(perreg["Scope"] == "south_korea") & (perreg["Model"] == "CLIP (per-region)") & (perreg["Top-k"] == 1)].iloc[0]
clip_sk3 = perreg[(perreg["Scope"] == "south_korea") & (perreg["Model"] == "CLIP (per-region)") & (perreg["Top-k"] == 3)].iloc[0]
rap_sk3 = main[(main["Scope"] == "south_korea") & (main["Model"] == "Proposed") & (main["Top-k"] == 3)].iloc[0]
app_g2 = (f"Table~\\ref{{tab:g2_full}} reports the per-region encoders on the same 4,943 confirmatory queries. These encoders are trained for one fixed region, so they fall outside the arbitrary-region setting of the main comparison and are reported here as an accuracy-oriented reference. "
          f"A per-region encoder has a lower mean than RaP-Weather in {n_pr_lower} shallow cells, namely {pr_lower_desc}. "
          f"After Holm correction, the significant per-region advantages are limited to {pr_sig_desc}. At Top-1 on {REGION_LABEL['south_korea']}, per-region CLIP attains {f4(clip_sk['Mean'])} versus {f4(rap_sk['Mean'])} for RaP-Weather (Holm $p{{=}}{fp(clip_sk['Holm p'])}$), and at Top-3 it attains {f4(clip_sk3['Mean'])} versus {f4(rap_sk3['Mean'])} (Holm $p{{=}}{fp(clip_sk3['Holm p'])}$). The remaining differences are not significant. "
          f"RaP-Weather has the lowest mean from Top-5 onward in every configuration and is significantly better than every per-region encoder at Top-50 and Top-100. A specialist trained for one fixed region can therefore improve single-best-match accuracy for that region, whereas the shared patch grid is stronger over deeper result lists and keeps training, storage, and query cost independent of the number of regions queried."
          )
app_g2 = app_g2.replace("$p{=}$", "$p{=}").replace("$$", "$")
write("txt_appendix_g2.tex", app_g2)

print("\nsummary:", {k: summary[k] for k in ("confirmatory_queries", "cells_rap_lowest_mean", "cells_rap_beats_all", "cells_rap_significant_loss", "visual_family_tests")})
