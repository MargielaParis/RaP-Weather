"""Device, seeding, CONFIG and path helpers.

Single source of truth for hyperparameters and paths. Every other module
imports CONFIG (or a copy with overrides) from here.

Paths resolve against the repository root, so scripts can run from any
working directory. Override the defaults with environment variables:

    RAP_DATA_DIR   directory holding the dataset files (default: <repo>/data)
    RAP_RUNS_DIR   directory for checkpoints and results (default: <repo>/runs)
"""
import os
import random
from pathlib import Path

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("RAP_DATA_DIR", REPO_ROOT / "data")).expanduser()
RUNS_DIR = Path(os.environ.get("RAP_RUNS_DIR", REPO_ROOT / "runs")).expanduser()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CONFIG = {
    "seed": 42,
    "seq_len": 6,                 # 6 frames x 3 h = 15 h of input
    "pred_step": 1,               # predict the next 3 h frame
    "subset_size": 25744,
    "batch_size": 64,
    "patch_emb_size": 8,          # D. The D-ablation overrides this with 4/16/32.
    "temperature_tau": 0.07,      # InfoNCE temperature (Appendix A)
    "alpha": 0.1,                 # weight of the InfoNCE term (Appendix A)
    "n_epochs": 10,               # same for RaP-Weather and every trained baseline
    "n_query": None,              # None = all 5,143 test-period candidate queries
    "top_k_list": [1, 3, 5, 10, 50, 100],
    "train_val_split": 0.8,       # chronological 80/20 split
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "temporal_dropout": 0.2,      # read by models.build_model()
    "lora_target_modules": ["qkv"],
    "paths": {
        "h5": str(DATA_DIR / "coms_25744_224.hdf5"),
        "weather_pkl": str(DATA_DIR / "coms_with_weather_dataset.pkl"),
        # Optional precomputed baseline embeddings; skipped when absent.
        "baseline_pkl": str(DATA_DIR / "precomputed_embeddings" / "baseline_embeddings.pkl"),
        # Fitted StandardScaler cache, created on first use.
        "weather_scaler": str(DATA_DIR / "weather_scaler.pkl"),
        "run_dir_template": str(RUNS_DIR / "f{patch_emb_size}" / "seq_{seq_len}_pred_1"),
    },
}
set_seed(CONFIG["seed"])


def run_dir(cfg=CONFIG):
    d = cfg["paths"]["run_dir_template"].format(**cfg)
    os.makedirs(d, exist_ok=True)
    return d


def model_save_path(cfg=CONFIG):
    return os.path.join(run_dir(cfg), "spatio_temporal_model.pth")


def embedding_save_path(cfg=CONFIG):
    return os.path.join(run_dir(cfg), "proposed_db_embeddings.pkl")


if __name__ == "__main__":
    print("Run directory for current CONFIG:", run_dir())
