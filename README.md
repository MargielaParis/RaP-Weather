# RaP-Weather

Code for the paper

> Ki-Yeon Park and Ha-Myung Park. **No Crop, No Recompute: Region-Aware Satellite Image Sequence Embeddings for Scalable Weather Retrieval.** *IEEE Access*, 2026.

RaP-Weather encodes each satellite image once into a grid of compact patch
embeddings (196 patches x D dimensions) and answers a query for any region by
selecting that region's patches. No cropping or re-encoding happens at query
time. The encoder is a ViT with LoRA adapters and an MLP bottleneck, trained
with a spatio-temporal prediction objective (predict the next 3-hour frame
from 6 past frames) combined with an InfoNCE term.

## Repository layout

| Path | Content |
|---|---|
| `rap_weather/config.py` | All hyperparameters and paths (single source of truth) |
| `rap_weather/models.py` | Patch auto-encoder, spatio-temporal forecaster, weather-vector loading and cleaning |
| `rap_weather/regions.py` | Query regions as patch indices and pixel boxes |
| `rap_weather/train.py`, `sanity_checks_and_save.py` | Training and retrieval-database construction |
| `rap_weather/eval_setup.py` | Chronological split, retrieval scores, bootstrap CIs (shared by every evaluation) |
| `rap_weather/eval_mse_retrieval.py` | Weather-observation MSE of retrieved images |
| `rap_weather/eval_image_quality.py` | LPIPS, PSNR, SSIM of retrieved images |
| `rap_weather/benchmark_query_time.py` | Query processing time |
| `rap_weather/baselines/` | Group 1 (off-the-shelf encoders), per-region fine-tuned encoders, unified comparison |
| `rap_weather/studies/` | D ablation, positional-encoding ablation, seasonal generalization, Pixel-NN, downsampling, storage cost |
| `scripts/` | Runnable experiments: Group 2 and full-dimension baselines, Pixel-NN per-query results, visual metrics, hyperparameter sweeps, single-image variant, figures |
| `analysis/` | Confirmatory statistics, LaTeX tables and paper figures from saved per-query results (no torch needed) |
| `results/` | Statistics CSVs behind the paper tables |
| `data/` | Expected dataset files (not distributed, see `data/README.md`) |

### Baseline names in code and paper

The comparison groups were renumbered in the paper after the code was
written. File names and result keys keep the code names.

| Paper | Code |
|---|---|
| Group 1 (off-the-shelf, frozen) | `rap_weather/baselines/group1_offshelf_baseline.py`, keys `... (Group1, off-the-shelf)` |
| Group 2 (trained on whole images, region cropped at query) | `scripts/group3_fulldim.py` (paper version), `scripts/group3_wholetrain_crop.py` (D=8 projection variant), keys `... (G3: whole-trained, crop@query)` and `... (Group2, whole-image fine-tuned)` for the full-image query |
| Per-region fine-tuned encoders (Appendix B) | `scripts/group2_fulldim_ablation.py` (paper version), `rap_weather/baselines/group2_finetuned_baseline.py`, keys `... (per-region)` |
| Pixel-NN | `scripts/pixel_nn_perquery.py`, `rap_weather/studies/baseline_comparison.py` |

## Setup

```bash
git clone <this repository>
cd RaP-Weather
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

Without the editable install, run the scripts with `PYTHONPATH=.` from the
repository root.

The paper experiments ran on one NVIDIA GeForce RTX 4090 (CUDA 11.8) with
PyTorch 2.6.0 and Python 3.12.2 on Linux. Training and the baseline stages
load the full image file into memory; plan for 32 GB of RAM or more.

### Data and output locations

Put the dataset files in `data/` (format in `data/README.md`) or set
environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `RAP_DATA_DIR` | `<repo>/data` | Dataset files |
| `RAP_RUNS_DIR` | `<repo>/runs` | Checkpoints, per-query results, figures. The main run dir is `<runs>/f8/seq_6_pred_1` |
| `RAP_STATS_DIR` | `<repo>/results` | Statistics CSVs and LaTeX tables |

## Reproducing the paper

All commands run from the repository root.

1. Train RaP-Weather and run the main evaluation suite.

   ```bash
   PYTHON=python scripts/run_experiments.sh repro train eval table1 speed dablation peablation
   ```

   `scripts/run_experiments.sh list` shows every stage. A stage skips models
   whose checkpoint already exists.

2. Train and evaluate the Group 2, per-region and Pixel-NN baselines used in
   the paper, and the visual metrics of Group 1 and Group 2.

   ```bash
   python scripts/group2_fulldim_ablation.py
   python scripts/group3_fulldim.py
   python scripts/pixel_nn_perquery.py
   python scripts/table1_group13.py
   ```

3. Compute the confirmatory statistics, tables and figures.

   ```bash
   python analysis/export_weather_vectors.py
   python analysis/statistical_audit.py
   python analysis/make_tables.py
   python scripts/regen_paper_figures.py
   ```

| Paper item | Produced by |
|---|---|
| Table 1 (implementation details) | `rap_weather/config.py`, `analysis/make_tables.py` |
| Table 2 (visual similarity) | stage `table1`, `scripts/table1_group13.py`, `analysis/statistical_audit.py` |
| Tables 3, 4 and Fig. 6 (weather-observation MSE) | stage `eval`, `scripts/group3_fulldim.py`, `scripts/pixel_nn_perquery.py`, `analysis/statistical_audit.py`, `analysis/plot_fig6.py` |
| Table 5 (design analyses: D, positional encoding, single-image variant) | stages `dablation`, `peablation`, `scripts/eval_single_image.py`, `analysis/make_tables.py` |
| Figs. 7, 8, 9 (single-image variant, D, positional encoding) | `scripts/regen_paper_figures.py` from the CSVs of the stages above |
| Figs. 2, 4, 5, 10 (satellite image panels) | `analysis/plot_fig2_3h.py`, `analysis/plot_fig45_qualitative.py`, `analysis/plot_fig10_failure.py` |
| Tables 7, 8 (Appendix A sweeps) | `scripts/sweep_retrieval_driver.py`, `scripts/sweep_dropout_driver.py`, `scripts/tau_resweep_d02.py`, `analysis/make_tables.py` |
| Table 9 (Appendix B, per-region encoders) | `scripts/group2_fulldim_ablation.py`, `analysis/statistical_audit.py`, `analysis/make_tables.py` |
| Per-image encoding latency (Sections V and VI, 2.5565 ms) | stage `speed` |

Table 6 (end-to-end query processing time against PSNR, SSIM and the
deep-learning pipelines) was measured with an earlier notebook version of the
pipeline and is not reproduced by this repository.

### Evaluation protocol

* Chronological split: the first 80% of the 25,744 images form the training
  set and the retrieval database (20,589 images after sequence construction);
  the last 20% provide 5,143 query candidates.
* 200 candidates (seed 123) were used for model selection and are excluded,
  leaving 4,943 confirmatory queries.
* Retrieval quality is the MSE between the standardized 112-dim surface
  observation vector of the query and those of the Top-k retrieved timestamps,
  for k in {1, 3, 5, 10, 50, 100}.
* RaP-Weather scores a region by the mean per-patch cosine similarity over the
  region's patches (`calculate_patch_mse_metrics`). Baselines use the cosine
  similarity of one global embedding (`calculate_mse_metrics`).
* Seed 42 for training, inference and bootstrap. One deterministic training
  run per configuration.

## Known differences between RaP-Weather and the trained baselines

These are reported in the paper (Table 1 and Discussion) and are kept as is
so that the released code matches the published numbers.

* Trained baselines use temporal dropout 0.1, RaP-Weather uses 0.2.
* Baseline backbones receive [0, 1] inputs without their own mean/std
  normalization.
* The single-image variant (Fig. 7) was trained with an earlier notebook
  version of the pipeline that is not included here. `scripts/eval_single_image.py`
  evaluates its checkpoint, given through `RAP_SINGLE_CKPT` and `RAP_SINGLE_PKL`.

## Citation

```bibtex
@article{park2026rapweather,
  title   = {No Crop, No Recompute: Region-Aware Satellite Image Sequence Embeddings for Scalable Weather Retrieval},
  author  = {Park, Ki-Yeon and Park, Ha-Myung},
  journal = {IEEE Access},
  year    = {2026}
}
```
