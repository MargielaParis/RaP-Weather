"""Phase-3 sweep of Appendix A: dropout, under the same validation-retrieval
protocol used for alpha/tau (phase 1) and LoRA rank/scale (phase 2).

Why two dropout axes: the model has two independent dropout knobs, and sweeping
only one would leave half the regularization pinned at its default:
  * lora_dropout        CONFIG["lora_dropout"], read by build_model()
  * temporal dropout    the nn.TransformerEncoderLayer of SpatioTemporalForecaster
                        (dropout, dropout1, dropout2 and self_attn.dropout),
                        default 0.1
The temporal dropout is overridden post-construction here, so every dropout
site is set explicitly without changing the shared model code. If a
non-default value wins, set it in CONFIG and retrain.

Protocol (identical to sweep_retrieval_driver, whose short_train/retrieval_score
this module reuses directly):
  3-epoch train (shuffle=True) -> encode DB -> 200 held-out validation queries
  (seed 123, disjoint from the 500 reported test queries) -> score = mean
  south_korea weather-MSE over Top-{1,5,10}. Lower is better.

alpha/tau/rank/scale are held at the FINAL selected config (CONFIG).

Outputs (run dir): dropout_grid_retrieval.csv (+ .partial.csv while
running, so an interrupted sweep resumes instead of restarting),
dropout_grid_retrieval_heatmap.png, best_dropout_retrieval.json.

Run from the repo root: python scripts/sweep_dropout_driver.py
"""
import json
import os

import pandas as pd
import torch

from rap_weather.config import CONFIG, run_dir
from rap_weather.sensitivity_sweep import SequenceDataset, _load_split_data
from torch.utils.data import DataLoader

# Reused wholesale so the dropout cells are trained/scored by the exact same code
# path as the alpha/tau and rank grids (module-level import also fixes the same
# VAL_TARGETS / seed 123 validation queries).
from sweep_retrieval_driver import (
    N_EPOCHS_SWEEP, N_VAL_QUERY, SCORE_KS, heatmap, retrieval_score, short_train,
)

LORA_DROPOUTS = (0.0, 0.05, 0.1, 0.2)
TEMPORAL_DROPOUTS = (0.0, 0.05, 0.1, 0.2)

OUT_CSV = "dropout_grid_retrieval.csv"
PARTIAL_CSV = "dropout_grid_retrieval.partial.csv"


def set_temporal_dropout(model, p):
    """Override every dropout site inside the temporal transformer.

    nn.TransformerEncoderLayer holds three nn.Dropout modules (dropout, dropout1,
    dropout2) plus a float `dropout` on its MultiheadAttention; missing any one
    of them would leave part of the regularization at the 0.1 default.
    """
    for layer in model.temporal_encoder.layers:
        layer.dropout.p = p
        layer.dropout1.p = p
        layer.dropout2.p = p
        layer.self_attn.dropout = p


def verify_dropout(model, lora_p, temporal_p):
    """Fail loudly rather than silently sweeping a knob that never moved."""
    for layer in model.temporal_encoder.layers:
        assert layer.dropout.p == temporal_p, f"temporal dropout not applied: {layer.dropout.p}"
        assert layer.dropout1.p == temporal_p and layer.dropout2.p == temporal_p
        assert layer.self_attn.dropout == temporal_p
    found = {m.p for n, m in model.vit.named_modules()
             if "lora_dropout" in n and isinstance(m, torch.nn.Dropout)}
    # PEFT uses nn.Identity instead of nn.Dropout when p == 0.0
    if lora_p > 0:
        assert found == {lora_p}, f"lora_dropout not applied: {found}"


def run_cell(lora_p, temporal_p, train_loader):
    model = short_train(
        CONFIG["alpha"], CONFIG["temperature_tau"], CONFIG["lora_r"], CONFIG["lora_alpha"],
        train_loader,
        cfg_extra={"lora_dropout": lora_p},
        post_build=lambda m: set_temporal_dropout(m, temporal_p),
        tag=f" ldo={lora_p} tdo={temporal_p}",
    )
    verify_dropout(model, lora_p, temporal_p)
    score, per_k = retrieval_score(model)
    del model
    torch.cuda.empty_cache()
    print(f"DSWEEP lora_dropout={lora_p} temporal_dropout={temporal_p} "
          f"val_retrieval_mse={score:.4f} (Top-1 {per_k[1]:.4f})", flush=True)
    return score, per_k


def main():
    d = run_dir(CONFIG)
    partial_path = os.path.join(d, PARTIAL_CSV)

    print(f"dropout sweep at FINAL alpha={CONFIG['alpha']} tau={CONFIG['temperature_tau']} "
          f"r={CONFIG['lora_r']} lora_alpha={CONFIG['lora_alpha']}")
    print(f"grid: lora_dropout {list(LORA_DROPOUTS)} x temporal_dropout {list(TEMPORAL_DROPOUTS)} "
          f"({len(LORA_DROPOUTS) * len(TEMPORAL_DROPOUTS)} cells)")
    print(f"score: {N_VAL_QUERY} val queries (seed 123), SK weather-MSE mean over "
          f"Top-{list(SCORE_KS)}, {N_EPOCHS_SWEEP}-epoch shuffle=True", flush=True)

    rows, done = [], set()
    if os.path.exists(partial_path):
        prev = pd.read_csv(partial_path)
        rows = prev.to_dict("records")
        done = {(round(r["lora_dropout"], 4), round(r["temporal_dropout"], 4)) for r in rows}
        print(f"resuming: {len(done)} cells already done in {PARTIAL_CSV}", flush=True)

    train_imgs, _, _, _ = _load_split_data(CONFIG)
    ds = SequenceDataset(train_imgs, seq_len=CONFIG["seq_len"], pred_step=CONFIG["pred_step"])
    train_loader = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=8)

    for ldo in LORA_DROPOUTS:
        for tdo in TEMPORAL_DROPOUTS:
            if (round(ldo, 4), round(tdo, 4)) in done:
                print(f"skip (done): lora_dropout={ldo} temporal_dropout={tdo}", flush=True)
                continue
            score, per_k = run_cell(ldo, tdo, train_loader)
            rows.append({"lora_dropout": ldo, "temporal_dropout": tdo,
                         "val_retrieval_mse": score,
                         **{f"Top-{k}": v for k, v in per_k.items()}})
            pd.DataFrame(rows).to_csv(partial_path, index=False)   # checkpoint each cell

    df = pd.DataFrame(rows).sort_values("val_retrieval_mse").reset_index(drop=True)
    df.to_csv(os.path.join(d, OUT_CSV), index=False)
    heatmap(df, "lora_dropout", "temporal_dropout",
            "Val retrieval MSE: LoRA x temporal dropout (task metric)",
            "dropout_grid_retrieval_heatmap.png")

    best_l = float(df.iloc[0]["lora_dropout"])
    best_t = float(df.iloc[0]["temporal_dropout"])
    best_score = float(df.iloc[0]["val_retrieval_mse"])
    with open(os.path.join(d, "best_dropout_retrieval.json"), "w") as f:
        json.dump({"lora_dropout": best_l, "temporal_dropout": best_t,
                   "val_retrieval_mse": best_score}, f, indent=2)

    # Score of the default setting, for comparison against the winner.
    cur = df[(df["lora_dropout"] == CONFIG["lora_dropout"]) & (df["temporal_dropout"] == 0.1)]
    print(f"\nDSWEEP-BEST lora_dropout={best_l} temporal_dropout={best_t} (score={best_score:.4f})")
    if not cur.empty:
        cur_score = float(cur.iloc[0]["val_retrieval_mse"])
        print(f"DSWEEP-CURRENT lora_dropout={CONFIG['lora_dropout']} temporal_dropout=0.1 "
              f"(score={cur_score:.4f}, rank {int(cur.index[0]) + 1}/{len(df)})")
        print(f"DSWEEP-DELTA best improves on current by {cur_score - best_score:.4f}")
        print("NOTE: compare this delta against seed-to-seed noise before adopting; "
              "the alpha/tau grid's top-2 cells were only 0.0009 apart.")
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"\nsaved: {os.path.join(d, OUT_CSV)}", flush=True)


if __name__ == "__main__":
    main()
