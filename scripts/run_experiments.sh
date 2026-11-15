#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh: RaP-Weather training and evaluation runner
#
# Usage (from any directory):
#   scripts/run_experiments.sh                 # default stages, in order
#   scripts/run_experiments.sh train eval      # only the named stages
#   scripts/run_experiments.sh list            # show available stages
#
# Stages (default order):
#   repro        reproducibility manifest -> <run_dir>/repro_config.json
#   docs         CPU-only notes: weather vector, downsampling, storage cost
#   train        main D=8 training + DB embeddings (skipped if the checkpoint
#                exists; FORCE_TRAIN=1 to retrain)
#   eval         full comparison in ONE process: RaP-Weather MSE eval ->
#                Group 1 -> per-region encoders -> Pixel-NN -> significance ->
#                seasonal generalization -> unified comparison
#   table1       LPIPS/PSNR/SSIM + significance
#   speed        query-time benchmark + hardware log
#   dablation    patch embedding size D in {4, 8, 16, 32}
#   peablation   positional-encoding subtraction ablation
#   diagnostics  PCA periodicity + mean-embedding trend
#   plots        result plots from saved CSVs
#   sweep        legacy alpha/tau/LoRA-rank sweep on embedding MSE; not in
#                the default run (the paper uses scripts/sweep_retrieval_driver.py)
#
# Env toggles:
#   FORCE_TRAIN=1   retrain main model even if checkpoint exists
#   FORCE_SWEEP=1   re-run sweep even though CSVs exist
#   SKIP_GROUP1=1 / SKIP_GROUP2=1 / SKIP_T7=1   trim the eval stage
#   PYTHON=...      python interpreter (default: python3)
#   RAP_DATA_DIR / RAP_RUNS_DIR   see rap_weather/config.py
#
# Notes:
#   * train / dablation / peablation / group2 load the full HDF5 into RAM
#     (~15 GB) and the weather pkl is ~10 GB: expect 32 GB+ RAM usage.
#   * The per-region stage trains 3 backbones x 4 regions = 12 models. Each
#     sub-step skips itself if its checkpoint already exists.
#   * The Group 2 and full-dimension baselines live in scripts/ (see README).
#   * Baseline backbones receive [0, 1] inputs without their own mean/std
#     normalization (reported in the paper, Table 1).
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg   # headless: modules call plt.show()

# Resolve the run dir from rap_weather.config so this guard and the Python side agree.
RUN_DIR="$("$PYTHON" -c 'from rap_weather.config import run_dir; print(run_dir())' 2>/dev/null | tail -n 1)"
CKPT="$RUN_DIR/spatio_temporal_model.pth"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR" "$RUN_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

log_run () {  # log_run <stage-name> <function>
    local name="$1"; shift
    local log="$LOG_DIR/${STAMP}_${name}.log"
    echo ""
    echo "===================================================================="
    echo "=== [$(date '+%F %T')] STAGE: $name"
    echo "===  log: $log"
    echo "===================================================================="
    if "$@" 2>&1 | tee "$log"; then
        echo "=== [$(date '+%F %T')] DONE: $name"
    else
        echo "!!! [$(date '+%F %T')] FAILED: $name (see $log)" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------- stages ----

stage_repro () {
    # Write the manifest into run_dir rather than the CWD.
    "$PYTHON" - <<'PY'
import json, os
from rap_weather.reproducibility import build_repro_config
from rap_weather.config import run_dir
path = os.path.join(run_dir(), "repro_config.json")
with open(path, "w") as f:
    json.dump(build_repro_config(), f, indent=2, default=str)
print(f"saved: {path}")
PY
}

stage_docs () {
    "$PYTHON" - <<'PY'
from rap_weather.studies.weather_vector_documentation import document_weather_vector
from rap_weather.studies.downsampling_justification import run_autocorrelation_analysis
from rap_weather.studies.storage_cost_analysis import compute_storage_cost
document_weather_vector()
run_autocorrelation_analysis()
compute_storage_cost()
PY
}

stage_train () {
    if [[ -f "$CKPT" && "${FORCE_TRAIN:-0}" != "1" ]]; then
        echo "checkpoint already exists: $CKPT"
        echo "skipping training (set FORCE_TRAIN=1 to retrain)"
        return 0
    fi
    # trains, sanity-checks, and saves model + DB embeddings + manifest
    "$PYTHON" -m rap_weather.sanity_checks_and_save
}

stage_eval () {
    # Everything here MUST share one Python process: Group1/Group2 merge their
    # results into eval_mse_retrieval.FINAL_REPORT* in memory, and ConvNeXt-FT /
    # significance / P-R-MAP / seasonal / unified all read from those dicts.
    "$PYTHON" - <<'PY'
import os

from rap_weather import eval_mse_retrieval as emr

# 1) Proposed model + precomputed-pkl baselines, all 4 regions
emr.run_evaluation()

# 2) Group 1 (off-the-shelf CLIP/ConvNeXt2/DINOv2, crop->encode)
if os.environ.get("SKIP_GROUP1") != "1":
    from rap_weather.baselines import group1_offshelf_baseline as g1
    g1.main()
else:
    print("SKIP_GROUP1=1: skipping Group 1")

# 3) Code Group 2: per-region predictive fine-tuned backbones (paper Appendix B;
#    12 trainings, ckpt-guarded)
if os.environ.get("SKIP_GROUP2") != "1":
    from rap_weather.baselines import group2_finetuned_baseline as g2
    g2.main()
else:
    print("SKIP_GROUP2=1: skipping Group 2")

# 4) Pixel-NN baseline + ConvNeXt-FT ablation (quantitative parts only; the
#    inspect_* figure helpers are interactive and skipped on purpose).
#    ConvNeXt-FT does not merge into FINAL_REPORT*: it is a backbone-swap
#    ablation, not a baseline, and writes convnext_ft_*.csv on its own. It runs
#    here because its table cites Proposed and ConvNeXt2-G2 raw values, which
#    only exist once steps 1-3 have populated FINAL_REPORT_RAW.
if os.environ.get("SKIP_T7") != "1":
    from rap_weather.studies import baseline_comparison as bc
    bc.run_pixel_nn_baseline()
    bc.train_finetuned_convnext_baseline()
    bc.evaluate_convnext_finetuned()
else:
    print("SKIP_T7=1: skipping pixel-NN / ConvNeXt-FT")

# 4b) Persist the retrieval rankings so seasonal / per-k re-analysis does not
#     need to rerun this (long) stage
emr.save_top_indices()

# 5) Significance: run AFTER all merges so the CSVs cover every registered
#    model, not just the pkl baselines
emr.run_significance()

# 6) Seasonal generalization (reads FINAL_REPORT_TOPIDX)
from rap_weather.studies.seasonal_generalization import run_seasonal_generalization
run_seasonal_generalization()

# 7) Unified comparison tables/plots (Proposed + Group 1 + trained baselines)
from rap_weather.baselines.unified_comparison import run_unified_mse_and_significance
run_unified_mse_and_significance()
PY
}

stage_table1 () {
    "$PYTHON" -m rap_weather.eval_image_quality
}

stage_speed () {
    "$PYTHON" -m rap_weather.benchmark_query_time
}

stage_dablation () {
    "$PYTHON" -m rap_weather.studies.d_ablation
}

stage_peablation () {
    "$PYTHON" -m rap_weather.studies.pe_ablation
}

stage_diagnostics () {
    "$PYTHON" -m rap_weather.embedding_diagnostics
}

stage_plots () {
    "$PYTHON" -m rap_weather.result_plots
}

stage_sweep () {
    if [[ -f "$RUN_DIR/alpha_tau_sweep.csv" && -f "$RUN_DIR/rank_scale_grid.csv" \
          && "${FORCE_SWEEP:-0}" != "1" ]]; then
        echo "sweep results already exist:"
        echo "  $RUN_DIR/alpha_tau_sweep.csv"
        echo "  $RUN_DIR/rank_scale_grid.csv"
        echo "skipping (set FORCE_SWEEP=1 to re-run; ~32 short trainings)"
        return 0
    fi
    "$PYTHON" -m rap_weather.sensitivity_sweep
}

# ------------------------------------------------------------- dispatcher ---

DEFAULT_STAGES=(repro docs train eval table1 speed dablation peablation diagnostics plots)
ALL_STAGES=(repro docs train eval table1 speed dablation peablation diagnostics plots sweep)

if [[ "${1:-}" == "list" ]]; then
    printf 'available stages:\n'
    printf '  %s\n' "${ALL_STAGES[@]}"
    printf 'default run: %s\n' "${DEFAULT_STAGES[*]}"
    exit 0
fi

STAGES=("${@:-}")
if [[ ${#STAGES[@]} -eq 0 || -z "${STAGES[0]}" ]]; then
    STAGES=("${DEFAULT_STAGES[@]}")
fi

for s in "${STAGES[@]}"; do
    if ! declare -F "stage_$s" > /dev/null; then
        echo "unknown stage: '$s' (try: scripts/run_experiments.sh list)" >&2
        exit 1
    fi
done

echo "RaP-Weather experiment runner"
echo "  root:    $ROOT"
echo "  run dir: $RUN_DIR"
echo "  python:  $($PYTHON --version 2>&1)"
if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
fi
echo "  stages:  ${STAGES[*]}"

for s in "${STAGES[@]}"; do
    log_run "$s" "stage_$s"
done

echo ""
echo "All requested stages finished. Outputs: $RUN_DIR | logs: $LOG_DIR"
