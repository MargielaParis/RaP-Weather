"""Single-image variant vs. RaP-Weather on south_korea (Section V, Fig. 9).

Evaluates a spatial-only encoder (trained without the temporal prediction
head) under the same protocol as eval_mse_retrieval.run_evaluation: all test
queries, DB = indices seq_len..SPLIT_IDX-1, query embedding from the image at
idx-1, patch-mean cosine retrieval, weather-observation MSE. RaP-Weather is
re-scored on the same queries for a paired Wilcoxon test.

Inputs (environment variables):
    RAP_SINGLE_CKPT  spatial encoder state_dict of the single-image variant
    RAP_SINGLE_PKL   its stored DB embeddings, used as a load sanity check
Outputs (run dir):
    single_vs_proposed_current_protocol_south_korea.csv
    single_vs_proposed_perquery_south_korea.npz
"""
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from scipy import stats

from rap_weather.config import CONFIG, RUNS_DIR, device, run_dir
from rap_weather.models import build_model, load_trained_model, normalize_image
from rap_weather.regions import region_tensor
from rap_weather.eval_setup import (
    TOP_K_LIST, SPLIT_IDX, target_indices, db_eval_dts, query_eval_dts,
    calculate_patch_mse_metrics, bootstrap_ci,
)

RUN = run_dir(CONFIG)
SINGLE_DIR = RUNS_DIR / "single_image"
SINGLE_CKPT = os.environ.get("RAP_SINGLE_CKPT", str(SINGLE_DIR / "spatial_encoder_final.pth"))
SINGLE_PKL = os.environ.get("RAP_SINGLE_PKL", str(SINGLE_DIR / "proposed_db_embeddings_single.pkl"))
REGION = 'south_korea'
BATCH = 64

# ---------------------------------------------------------------- single model
single = build_model(CONFIG, pretrained_vit=False, device=device)
sd = torch.load(SINGLE_CKPT, map_location=device, weights_only=True)
res = single.load_state_dict(sd, strict=False)
print('missing (expected: temporal/pred_head only):', res.missing_keys)
print('unexpected:', res.unexpected_keys)
assert not res.unexpected_keys, 'checkpoint has keys the current architecture lacks'
assert all(k.startswith(('temporal_pos_embed', 'temporal_encoder.', 'pred_head.'))
           for k in res.missing_keys), 'spatial weights missing from checkpoint'
single.eval()

# sanity: reproduce a slice of the stored DB embeddings with the loaded encoder
with open(SINGLE_PKL, 'rb') as f:
    ref = pickle.load(f)
ref_emb = ref['embeddings']  # [20594, 196, 8], h5 index i -> row i
with h5py.File(CONFIG['paths']['h5'], 'r') as f:
    imgs = torch.tensor(f['images'][6:6 + BATCH], dtype=torch.float32)
with torch.no_grad():
    got = single.forward_spatial(normalize_image(imgs).to(device)).cpu()
cos = F.cosine_similarity(got.reshape(-1, got.shape[-1]),
                          ref_emb[6:6 + BATCH].reshape(-1, got.shape[-1]), dim=1)
print(f'sanity cos vs stored embeddings: mean={cos.mean():.6f} min={cos.min():.6f}')
assert cos.mean() > 0.999, 'loaded encoder does not reproduce stored embeddings'

curr_indices = region_tensor(REGION, device)


def extract(model):
    db, q = [], []
    with h5py.File(CONFIG['paths']['h5'], 'r') as f:
        dset = f['images']
        with torch.no_grad():
            for s in range(CONFIG['seq_len'], SPLIT_IDX, BATCH):
                e = min(s + BATCH, SPLIT_IDX)
                t = normalize_image(torch.tensor(dset[s:e], dtype=torch.float32)).to(device)
                db.append(model.forward_spatial(t).index_select(1, curr_indices).cpu())
            qidx = np.sort(target_indices - 1)
            assert (qidx == target_indices - 1).all(), 'target_indices not sorted'
            for s in range(0, len(qidx), BATCH):
                batch_idx = qidx[s:s + BATCH]
                t = normalize_image(torch.tensor(dset[batch_idx], dtype=torch.float32)).to(device)
                q.append(model.forward_spatial(t).index_select(1, curr_indices).cpu())
    return torch.cat(db).to(device), torch.cat(q).to(device)


results = {}
for name, model in (('RaP-Single', single), ('Proposed', None)):
    if model is None:
        model = load_trained_model(checkpoint_path=f'{RUN}/spatio_temporal_model.pth',
                                   cfg=CONFIG, device=device)
    print(f'--- extracting {name}')
    db, q = extract(model)
    mean_res, raw_res, _ = calculate_patch_mse_metrics(q, db, query_eval_dts, db_eval_dts)
    results[name] = (mean_res, raw_res)
    print(name, {k: round(v, 4) for k, v in mean_res.items()})
    del db, q, model
    torch.cuda.empty_cache()

# ------------------------------------------------------------------ report
rows = []
for k in TOP_K_LIST:
    s_raw = np.array(results['RaP-Single'][1][k])
    p_raw = np.array(results['Proposed'][1][k])
    assert len(s_raw) == len(p_raw)
    sm, slo, shi = bootstrap_ci(s_raw)
    pm, plo, phi = bootstrap_ci(p_raw)
    w = stats.wilcoxon(s_raw, p_raw)
    rows.append({'Top-k': k,
                 'Proposed MSE': pm, 'Proposed CI Low': plo, 'Proposed CI High': phi,
                 'RaP-Single MSE': sm, 'Single CI Low': slo, 'Single CI High': shi,
                 'N': len(s_raw), 'Wilcoxon p': w.pvalue})

import pandas as pd
df = pd.DataFrame(rows)
out = f'{RUN}/single_vs_proposed_current_protocol_{REGION}.csv'
df.to_csv(out, index=False)
np.savez(f'{RUN}/single_vs_proposed_perquery_{REGION}.npz',
         **{f'single_k{k}': np.array(results['RaP-Single'][1][k]) for k in TOP_K_LIST},
         **{f'proposed_k{k}': np.array(results['Proposed'][1][k]) for k in TOP_K_LIST})
print(df.to_string(index=False))
print('saved', out)
