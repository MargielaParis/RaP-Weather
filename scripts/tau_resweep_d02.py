"""Appendix A tau check at temporal dropout 0.2.

Background: tau=0.07 (config.py) was selected by a 200-query validation sweep at
temporal dropout 0.1 and then confirmed on the full holdout. This script reselects
tau on the validation set only, with temporal dropout 0.2. All other
hyperparameters are fixed to CONFIG.

Protocol: same as sweep_retrieval_driver (3-epoch short training, SK region, mean
weather-MSE over Top-1/5/10 on 200 validation queries). The validation queries are
rebuilt with the same recipe as reconstruct_selection_queries() in
analysis/statistical_audit.py (seed 42 -> exclude 500 legacy queries, seed 123 -> 200),
and the candidate pool is asserted to match.

Output: $TAU_RESWEEP_OUT (default <runs>/tau_resweep_d02) receives
tau_resweep_d02_retrieval.csv (+ .partial.csv for resuming) and best_tau_d02.json.
Never writes into the main run dir.

Run from the repo root: python scripts/tau_resweep_d02.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sweep_retrieval_driver as srd
from sweep_dropout_driver import set_temporal_dropout, verify_dropout
from rap_weather.config import CONFIG, RUNS_DIR, run_dir
from rap_weather.eval_setup import candidate_indices, datetimes_full
from rap_weather.sensitivity_sweep import SequenceDataset, _load_split_data

TAUS = (0.05, 0.07, 0.1, 0.2)
ALPHA = 0.1
LORA_R, LORA_ALPHA = 16, 32
TEMPORAL_DROPOUT = 0.2
OUT_DIR = os.environ.get("TAU_RESWEEP_OUT", str(RUNS_DIR / "tau_resweep_d02"))
OUT_CSV = os.path.join(OUT_DIR, "tau_resweep_d02_retrieval.csv")
PARTIAL_CSV = os.path.join(OUT_DIR, "tau_resweep_d02_retrieval.partial.csv")


def reconstruct_selection_queries():
    """Same recipe as reconstruct_selection_queries() in analysis/statistical_audit.py."""
    n_total, seq_len = 25_744, 6
    split = int(n_total * 0.8)
    cands = np.arange(split + seq_len, n_total)
    assert np.array_equal(cands, candidate_indices), "candidate_indices mismatch vs audit recipe"
    legacy = np.random.RandomState(42).choice(cands, 500, replace=False)
    pool = np.setdiff1d(cands, legacy)
    sel = np.random.RandomState(123).choice(pool, 200, replace=False)
    assert len(sel) == 200 and np.all(np.isin(sel, cands))
    return np.sort(sel)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    assert os.path.abspath(OUT_DIR) != os.path.abspath(run_dir(CONFIG)), "must not write into the main run dir"

    # --- validation queries: override the module-level set in the driver ---
    val = reconstruct_selection_queries()
    srd.VAL_TARGETS = val
    srd.VAL_QUERY_DTS = [datetimes_full[i - 1] for i in val]
    assert len(srd.VAL_TARGETS) == srd.N_VAL_QUERY == 200

    print(f"CONFIG: alpha={ALPHA} lora_r={LORA_R} lora_alpha={LORA_ALPHA} "
          f"temporal_dropout={TEMPORAL_DROPOUT} (CONFIG has {CONFIG.get('temporal_dropout')}) "
          f"| taus={TAUS} | {srd.N_EPOCHS_SWEEP} epochs | score=SK Top-{list(srd.SCORE_KS)} mean",
          flush=True)

    train_imgs, _, _, _ = _load_split_data(CONFIG)
    ds = SequenceDataset(train_imgs, seq_len=CONFIG["seq_len"], pred_step=CONFIG["pred_step"])
    train_loader = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=8)

    done = {}
    if os.path.exists(PARTIAL_CSV):
        for _, r in pd.read_csv(PARTIAL_CSV).iterrows():
            done[float(r["tau"])] = r.to_dict()
        print(f"resuming: {sorted(done)} already done", flush=True)

    rows = list(done.values())
    for tau in TAUS:
        if tau in done:
            continue
        t0 = time.time()

        def post_build(m):
            set_temporal_dropout(m, TEMPORAL_DROPOUT)
            verify_dropout(m, CONFIG["lora_dropout"], TEMPORAL_DROPOUT)

        model = srd.short_train(ALPHA, tau, LORA_R, LORA_ALPHA, train_loader,
                                cfg_extra={"temporal_dropout": TEMPORAL_DROPOUT},
                                post_build=post_build, tag=f" d{TEMPORAL_DROPOUT}")
        score, per_k = srd.retrieval_score(model)
        row = {"alpha": ALPHA, "tau": tau, "temporal_dropout": TEMPORAL_DROPOUT,
               "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
               "val_retrieval_mse": score, **{f"Top-{k}": v for k, v in per_k.items()},
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(PARTIAL_CSV, index=False)
        print(f"tau={tau}: val_retrieval_mse={score:.4f} per_k={per_k} ({row['seconds']}s)", flush=True)
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values("val_retrieval_mse").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    best = df.iloc[0]
    meta = {
        "best_tau": float(best["tau"]),
        "best_val_retrieval_mse": float(best["val_retrieval_mse"]),
        "n_val_query": 200,
        "val_query_recipe": "seed42->legacy500 excluded, seed123->200 (== analysis/statistical_audit.py)",
        "temporal_dropout": TEMPORAL_DROPOUT,
        "n_epochs_sweep": srd.N_EPOCHS_SWEEP,
        "score": f"SK weather-MSE mean over Top-{list(srd.SCORE_KS)}",
        "csv": OUT_CSV,
    }
    with open(os.path.join(OUT_DIR, "best_tau_d02.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\n=== tau re-sweep @ dropout 0.2 ===")
    print(df.to_string(index=False))
    print(f"\nbest tau = {meta['best_tau']} (val {meta['best_val_retrieval_mse']:.4f})")


if __name__ == "__main__":
    main()
