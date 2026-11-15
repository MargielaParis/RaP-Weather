"""RaP-Weather: region-aware satellite image sequence embeddings for weather retrieval.

Import order matters for a couple of modules that compute module-level state at
import time (config.py sets the random seed; eval_setup.py derives the
deterministic query/DB split used by most evaluation modules).

Layout:
    config.py                  device, seeding, CONFIG, path helpers
    regions.py                 SK_1 / SK_2 / low_left / total regions
    models.py                  model classes and shared utilities
    reproducibility.py         reproducibility manifest
    sensitivity_sweep.py       legacy embedding-MSE alpha/tau/LoRA-rank sweep;
                               also provides SequenceDataset
    train.py                   training and search-DB build
    sanity_checks_and_save.py  debug checks and artifact saving
    eval_setup.py              shared evaluation infrastructure
    eval_mse_retrieval.py      weather MSE, retrieval quality, significance tests
    eval_image_quality.py      LPIPS/PSNR/SSIM (Table 1)
    benchmark_query_time.py    query-time benchmark (Table 2)
    result_plots.py            plots from saved CSVs
    embedding_diagnostics.py   PCA periodicity and mean-value trend

    baselines/                 Group 1 off-the-shelf encoders, per-region
                               fine-tuned encoders, unified comparison
    studies/                   D and PE ablations, seasonal generalization,
                               Pixel-NN, downsampling, storage cost,
                               weather-vector documentation
"""
