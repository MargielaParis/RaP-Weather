"""Reproducibility manifest: configuration, implementation details, and
hardware/software environment of a run."""
import datetime
import json
import platform
import subprocess

import torch

from .config import CONFIG


def get_gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unavailable"


def build_repro_config(cfg=CONFIG):
    return {
        **{f"data.{k}": v for k, v in cfg.items() if k != "paths"},
        "data.paths": cfg["paths"],
        "vit_backbone": "vit_base_patch16_224 (timm, ImageNet-pretrained)",
        "image_normalization": "normalize_image(): [0,255] -> [0,1] (same for every model, see models.normalize_image)",
        "weather_vector_missing_data": (
            "The source pickle stores KMA missing-data codes as ordinary numbers, so a NaN "
            "check passes them through. The final robust per-column rule detects 488 values "
            "over 441 of 25,744 samples (1.71%) across 35 columns. "
            "Left in place they inflate the fitted std of the pressure variables by 4.1-4.6x, "
            "which suppresses that variable group in the weather-MSE the retrieval is scored on. "
            "models.clean_weather_vectors() flags them by robust z-score "
            "(|x-median|/(1.4826*MAD) > 20, or value <= -99) and imputes the column median "
            "before StandardScaler is fitted. -9.9 is NOT treated as a code: in the dewpoint and "
            "temperature blocks it occurs at 0.9-1.1x the frequency of its neighbours, i.e. it is "
            "real winter data."),
        "mlp_bottleneck_dims": [768, 512, 256, 128, cfg["patch_emb_size"]],
        "temporal_num_layers": 2, "temporal_num_heads": 4, "temporal_dim_feedforward": 128,
        "temporal_dropout": cfg.get("temporal_dropout", 0.1), "temporal_norm_first": True,
        "optimizer": "AdamW", "lr_backbone_and_bottleneck": 5e-6, "lr_temporal_head": 3e-4,
        "grad_clip_max_norm": 1.0, "mixed_precision": "torch.amp autocast + GradScaler",
        "alpha_tau_selection": ("selected by validation retrieval performance: 4x4 grid "
                                "over alpha in {0.05,0.1,0.5,1.0} x tau in {0.05,0.07,0.1,0.2}, each cell "
                                "trained 3 epochs and scored by mean Top-1/5/10 weather-MSE on 200 held-out "
                                "south_korea selection queries (seed 123), which analysis/statistical_audit.py "
                                "excludes from the 4,943 confirmatory queries. Winner "
                                "alpha=0.1, tau=0.07 (score 1.0551, Top-1 0.8837); the "
                                "previous default tau=0.1 ranked 3rd (1.0675). See "
                                "alpha_tau_sweep_retrieval.csv / best_hparams_retrieval.json."),
        "lora_rank_selection": ("selected by the same validation-retrieval protocol: rank in {4,8,16,32} x "
                                "scale-ratio in {0.5,1,2,4}, evaluated at the winning alpha/tau. Winner "
                                "r=16, lora_alpha=32 (ratio 2, score 1.0551) vs ratio 4 (1.1768); the r=16/"
                                "lora_alpha=32 cell reproduced the phase-1 value exactly (determinism "
                                "check). See rank_scale_grid_retrieval.csv."),
        "n_training_runs": "one deterministic run per configuration (seed 42)",
        "gpu": get_gpu_name(), "cuda_version": torch.version.cuda, "torch_version": torch.__version__,
        "python_version": platform.python_version(), "os": platform.platform(),
        "cpu": platform.processor() or "unavailable", "git_commit": get_git_commit(),
        "manifest_generated_at": datetime.datetime.now().isoformat(),
    }


if __name__ == "__main__":
    REPRO_CONFIG = build_repro_config()
    print("Reproducibility Config:")
    for k, v in REPRO_CONFIG.items():
        print(f"   - {k}: {v}")
    with open("repro_config.json", "w") as f:
        json.dump(REPRO_CONFIG, f, indent=2, default=str)
    print("\nSaved to repro_config.json")
