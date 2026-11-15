"""Model definitions and shared utilities: encoder, checkpoint loading, and the
weather-vector map used for retrieval scoring."""
import gc
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import timm
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, device, model_save_path


class PatchAutoEncoder(nn.Module):
    """768-dim ViT patch feature -> D-dim compressed embedding."""
    def __init__(self, patch_emb_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, patch_emb_size)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        return self.encoder(x)


class SpatioTemporalForecaster(nn.Module):
    """ViT (+ LoRA) -> PatchAutoEncoder -> temporal Transformer -> next-step prediction."""
    def __init__(self, vit_model, ae_model, seq_len, embed_dim, temporal_dropout=0.1):
        super().__init__()
        self.vit = vit_model
        self.ae = ae_model
        self.temporal_dropout = temporal_dropout
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=128,
            batch_first=True, dropout=temporal_dropout, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.pred_head = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.pred_head.weight)
        nn.init.zeros_(self.pred_head.bias)

    def forward_spatial(self, img):
        """Single-image -> (B, 196, D) spatial embedding. Used for BOTH training
        (ground-truth target encoding) and inference (DB / query encoding)."""
        feats = self.vit.forward_features(img)[:, 1:, :]
        feats = feats - self.vit.pos_embed[:, 1:, :]   # subtract absolute PE (paper III-A)
        return self.ae.encode(feats)

    def forward(self, seq_input):
        """(B, T, C, H, W) sequence -> (B, 196, D) predicted t+1 embedding."""
        B, T, C, H, W = seq_input.shape
        flat_in = seq_input.view(B * T, C, H, W)
        flat_embs = self.forward_spatial(flat_in)
        _, N, D = flat_embs.shape
        embs = flat_embs.view(B, T, N, D).permute(0, 2, 1, 3).reshape(B * N, T, D)
        embs = embs + self.temporal_pos_embed
        t_out = self.temporal_encoder(embs)
        pred = self.pred_head(t_out[:, -1, :])
        return pred.view(B, N, D)


def normalize_image(img_tensor):
    """Image normalization shared by training and evaluation.

    Assumes raw inputs in [0, 255] and maps them to [0, 1] with a fixed /255
    (no per-batch min/max), so train-time and eval-time inputs are scaled
    identically. Inputs already in [0, 1] are returned unchanged."""
    img_tensor = img_tensor.float()
    if img_tensor.max() > 1.5:  # heuristic: still in raw [0, 255]-ish range
        img_tensor = img_tensor / 255.0
    return img_tensor


def build_model(cfg=CONFIG, pretrained_vit=True, device=device):
    base_vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained_vit)
    base_vit.reset_classifier(0)
    lora_config = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias="none", task_type=TaskType.FEATURE_EXTRACTION, target_modules=cfg["lora_target_modules"],
    )
    vit = get_peft_model(base_vit, lora_config).to(device)
    ae = PatchAutoEncoder(cfg["patch_emb_size"]).to(device)
    model = SpatioTemporalForecaster(vit, ae, seq_len=cfg["seq_len"],
                                    embed_dim=cfg["patch_emb_size"],
                                    temporal_dropout=cfg.get("temporal_dropout", 0.1)).to(device)
    return model


def load_trained_model(checkpoint_path=None, cfg=CONFIG, device=device, strict=True):
    """Build the model and load a checkpoint. strict=True by default; with
    strict=False any missing/unexpected keys are printed so a partial load is
    never silent."""
    model = build_model(cfg, pretrained_vit=False, device=device)
    ckpt_path = checkpoint_path or model_save_path(cfg)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    result = model.load_state_dict(state, strict=strict)
    if not strict and result is not None:
        if result.missing_keys:
            print(f"Missing keys on load: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"Unexpected keys on load: {result.unexpected_keys}")
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    return model


_WEATHER_CACHE = {}

# Missing-data codes survive in the source pickle as ordinary numbers, so a NaN
# check passes them through. Measured on coms_with_weather_dataset.pkl:
#   485x  100.0   in the 32 pressure variables (isolated spike: 99.0/99.5/100.5
#                 /101.0 all occur zero times, and 100 hPa is not a surface value)
#     1x  888.0   in a pressure variable
#     2x -998.0
#     1x  -99.8
# Left untouched they inflate the fitted std of the pressure variables by 4.1-4.6x,
# which squashes that whole variable group's contribution to the weather-MSE the
# retrieval is scored on. Note -9.9 is NOT a code here: in the dewpoint and
# temperature blocks it occurs at 0.9-1.1x the frequency of its neighbours, i.e.
# it is real winter data.
WEATHER_SENTINEL_ROBUST_Z = 20.0   # |x - median| / (1.4826 * MAD)
WEATHER_SENTINEL_FLOOR = -99.0     # KMA-style -999/-99.x codes


def find_weather_sentinels(raw_vecs):
    """Boolean mask of missing-data codes, per column, without hardcoding which
    column is which variable. The robust z-score keeps 100.0 flagged in pressure
    (median ~1013) while leaving a legitimate 100.0 alone in a humidity column
    (median ~73); a fixed value list could not tell those apart."""
    med = np.median(raw_vecs, axis=0)
    mad = np.median(np.abs(raw_vecs - med), axis=0)
    scale = np.where(mad > 0, 1.4826 * mad, 1.0)
    robust_z = np.abs(raw_vecs - med) / scale
    return (robust_z > WEATHER_SENTINEL_ROBUST_Z) | (raw_vecs <= WEATHER_SENTINEL_FLOOR)


def clean_weather_vectors(raw_vecs, verbose=True):
    """Replace missing-data codes with the column median of the valid values.

    Median rather than mean: the codes are extreme enough that a mean computed
    over the contaminated column is itself shifted. Imputing keeps every sample
    in the evaluation, which matters because dropping them would change the
    query count and break comparability with earlier runs."""
    mask = find_weather_sentinels(raw_vecs)
    if not mask.any():
        if verbose:
            print("weather vectors: no sentinel codes found")
        return raw_vecs.copy(), mask

    cleaned = raw_vecs.astype(np.float64, copy=True)
    cleaned[mask] = np.nan
    col_median = np.nanmedian(cleaned, axis=0)
    idx = np.where(mask)
    cleaned[idx] = np.take(col_median, idx[1])

    if verbose:
        n_rows = int(mask.any(axis=1).sum())
        print(f"weather vectors: imputed {int(mask.sum())} sentinel values "
              f"across {n_rows} / {len(raw_vecs)} samples "
              f"({n_rows / len(raw_vecs) * 100:.2f}%), {int(mask.any(axis=0).sum())} columns affected")
    return cleaned, mask


def get_weather_map_and_scaler(cfg=CONFIG, force_refit=False):
    """Return (datetime -> standardized weather vector, fitted StandardScaler).

    The scaler is fitted once and cached on disk and in memory, so every
    evaluation uses the same standardization."""
    pkl_path = cfg["paths"]["weather_pkl"]
    if pkl_path in _WEATHER_CACHE and not force_refit:
        return _WEATHER_CACHE[pkl_path]

    with open(pkl_path, "rb") as f:
        gt_data = pickle.load(f)
    raw_vecs = np.array([item[2] for item in gt_data])
    dts = [item[0] for item in gt_data]

    # Must happen before BOTH the fit and the transform branch below: a scaler
    # fitted on clean data would still be handed contaminated inputs otherwise.
    raw_vecs, _ = clean_weather_vectors(raw_vecs)

    scaler_path = cfg["paths"]["weather_scaler"]
    if os.path.exists(scaler_path) and not force_refit:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        scaled_vecs = scaler.transform(raw_vecs)
    else:
        scaler = StandardScaler()
        scaled_vecs = scaler.fit_transform(raw_vecs)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"Fitted and cached weather scaler: {scaler_path}")

    weather_map = {dt: vec for dt, vec in zip(dts, scaled_vecs)}
    _WEATHER_CACHE[pkl_path] = (weather_map, scaler)
    return weather_map, scaler


def cleanup_gpu(*freed_labels):
    """Run the garbage collector and empty the CUDA cache.

    Callers must `del` their own references before calling this (deleting
    from a `locals()` snapshot would not drop the real binding);
    `freed_labels` is only used for the log line."""
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Freed: {list(freed_labels)}")
    print(f"   Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB | "
          f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
