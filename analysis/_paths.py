"""Path layout shared by the analysis scripts.

Mirrors rap_weather.config without importing torch. Environment overrides:
RAP_DATA_DIR, RAP_RUNS_DIR (same meaning as in rap_weather.config) and
RAP_STATS_DIR (statistics CSVs, default: <repo>/results).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("RAP_DATA_DIR", REPO_ROOT / "data")).expanduser()
RUNS_DIR = Path(os.environ.get("RAP_RUNS_DIR", REPO_ROOT / "runs")).expanduser()
RUN = RUNS_DIR / "f8" / "seq_6_pred_1"
STATS_DIR = Path(os.environ.get("RAP_STATS_DIR", REPO_ROOT / "results")).expanduser()
TABLES_DIR = STATS_DIR / "tables"
FIGURES_DIR = RUNS_DIR / "figures"
H5 = DATA_DIR / "coms_25744_224.hdf5"
WEATHER_NPZ = DATA_DIR / "weather_vectors_raw.npz"
