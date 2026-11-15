"""Paper "Group 2", D=8 projection variant: backbone trained on whole images, region cropped at query time.

Code name "group3" corresponds to the paper's "Group 2"; code "group2" (G2) is the
paper's per-region encoders. This is the D=8 projection variant; group3_fulldim.py
is the full-dimension variant used in the final paper.

Fills the empty cell of the experimental design:

                  trained on crops           trained on whole image
    crop query    G2 per-region (12 models)  G3  <-- here
    patch query   -                          Proposed (1 model)

G2 trains a dedicated model per region (3 backbones x 4 regions = 12), which is
expensive. G3 tests the most realistic naive alternative: "train once on whole
images and crop at query time". No training is done; the existing
group2_<backbone>_total_patches checkpoints (box (0,0,224,224), i.e. trained on the
whole image) are reused and only the evaluation box is set to the target region.

The key issue is the train/test domain mismatch: the model is trained only on full
224 scenes, but at inference it receives a region crop upscaled to 224.

Sanity check: with region_key="total_patches" both the checkpoint and the box are
identical to G2, so the result must match group2_finetuned_total_patches.csv.
A mismatch means the loading or evaluation path in this script is wrong.

Proposed, G2 (same region) and G3 are evaluated in one process so that per-query
raw values are kept and paired Wilcoxon tests can be computed (the main evaluation
pipeline does not save raw values, so post hoc paired tests are not possible there).

Run from the repo root: python scripts/group3_wholetrain_crop.py
"""
import json
import os

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from tqdm import tqdm

from rap_weather.config import CONFIG, device, run_dir
from rap_weather.models import load_trained_model, normalize_image
from rap_weather.regions import REGIONS, region_tensor
from rap_weather.eval_setup import (
    bootstrap_ci, calculate_mse_metrics, calculate_patch_mse_metrics,
    db_eval_dts, db_indices, query_eval_dts, target_indices,
)
from rap_weather.baselines.backbone_registry import (
    BACKBONE_SPECS, crop_and_pad,
)
from rap_weather.baselines.group2_finetuned_baseline import (
    WholeRegionPredictor, group2_checkpoint_path,
)

# Padding variants of Group 3. crop_and_resize blows the 48x64 south_korea box up
# to 224 (4.7x wide, 3.5x tall, aspect ratio changed), so a G3 failure mixes two
# causes: the lost surrounding context, and a scale the model never trained on.
# These conditions keep the crop at native resolution so the two can be told apart.
PAD_MODES = ("mean", "zero")

REGIONS_TO_RUN = ["south_korea"]      # headline region in the paper
SELFCHECK_REGION = "total_patches"    # sanity check (must reproduce G2)
TRAINED_ON = "total_patches"          # checkpoint trained on the whole image


def _encode_backbone(model_g, box, indices, desc):
    """h5py fancy indexing requires increasing indices -> sort, then restore the order (as in G2)."""
    indices = np.asarray(indices)
    order = np.argsort(indices)
    sorted_indices = indices[order]
    out = []
    with h5py.File(CONFIG["paths"]["h5"], "r") as f, torch.no_grad():
        dset = f["images"]
        for start in tqdm(range(0, len(sorted_indices), CONFIG["batch_size"]),
                          desc=desc, leave=False):
            batch_idx = sorted_indices[start:start + CONFIG["batch_size"]]
            imgs = normalize_image(torch.tensor(dset[list(batch_idx)], dtype=torch.float32)).to(device)
            out.append(model_g.forward_spatial(imgs, box).cpu())
    return torch.cat(out, dim=0)[np.argsort(order)]


def _encode_backbone_pad(model_g, box, indices, desc, pad_mode):
    """Same as _encode_backbone but pads instead of resizing. Mirrors
    WholeRegionPredictor.forward_spatial() with only the crop transform swapped,
    so the shared class stays untouched (G2 must keep using resize)."""
    indices = np.asarray(indices)
    order = np.argsort(indices)
    sorted_indices = indices[order]
    out = []
    with h5py.File(CONFIG["paths"]["h5"], "r") as f, torch.no_grad():
        dset = f["images"]
        for start in tqdm(range(0, len(sorted_indices), CONFIG["batch_size"]),
                          desc=desc, leave=False):
            batch_idx = sorted_indices[start:start + CONFIG["batch_size"]]
            imgs = normalize_image(torch.tensor(dset[list(batch_idx)], dtype=torch.float32)).to(device)
            crop = crop_and_pad(imgs, box, model_g.input_size, pad_mode)
            out.append(model_g.proj(model_g.backbone(crop)).cpu())
    return torch.cat(out, dim=0)[np.argsort(order)]


def eval_backbone_pad(backbone_name, ckpt_region, eval_region, pad_mode):
    """Group 3 with the crop kept at native resolution."""
    ckpt = group2_checkpoint_path(backbone_name, ckpt_region, CONFIG)
    if not os.path.exists(ckpt):
        print(f"  checkpoint not found, skipping: {ckpt}")
        return None, None
    box = REGIONS[eval_region]["box"]
    m = WholeRegionPredictor(backbone_name, embed_dim=CONFIG["patch_emb_size"],
                             seq_len=CONFIG["seq_len"]).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    tag = f"[{backbone_name}|pad={pad_mode}]"
    db = _encode_backbone_pad(m, box, db_indices, f"{tag} DB", pad_mode)
    q = _encode_backbone_pad(m, box, target_indices - 1, f"{tag} Q", pad_mode)
    mean_res, raw_res, _ = calculate_mse_metrics(q.to(device), db.to(device), query_eval_dts, db_eval_dts)
    del m, db, q
    torch.cuda.empty_cache()
    return mean_res, raw_res


def eval_backbone(backbone_name, ckpt_region, eval_region):
    """Evaluate a backbone trained on ckpt_region using the eval_region box."""
    ckpt = group2_checkpoint_path(backbone_name, ckpt_region, CONFIG)
    if not os.path.exists(ckpt):
        print(f"  checkpoint not found, skipping: {ckpt}")
        return None, None
    box = REGIONS[eval_region]["box"]
    m = WholeRegionPredictor(backbone_name, embed_dim=CONFIG["patch_emb_size"],
                             seq_len=CONFIG["seq_len"]).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    db = _encode_backbone(m, box, db_indices, f"[{backbone_name}|ckpt={ckpt_region}->{eval_region}] DB")
    q = _encode_backbone(m, box, target_indices - 1, f"[{backbone_name}|ckpt={ckpt_region}->{eval_region}] Q")
    mean_res, raw_res, _ = calculate_mse_metrics(q.to(device), db.to(device), query_eval_dts, db_eval_dts)
    del m, db, q
    torch.cuda.empty_cache()
    return mean_res, raw_res


def eval_proposed(region_key):
    """Proposed: encode the whole image once -> select the region's patch indices (no crop)."""
    model = load_trained_model(cfg=CONFIG, device=device)
    model.eval()
    idx = region_tensor(region_key, torch.device("cpu")).to(device)

    def enc(indices, desc):
        indices = np.asarray(indices)
        order = np.argsort(indices)
        s = indices[order]
        out = []
        with h5py.File(CONFIG["paths"]["h5"], "r") as f, torch.no_grad():
            dset = f["images"]
            for start in tqdm(range(0, len(s), CONFIG["batch_size"]), desc=desc, leave=False):
                b = s[start:start + CONFIG["batch_size"]]
                imgs = normalize_image(torch.tensor(dset[list(b)], dtype=torch.float32)).to(device)
                out.append(model.forward_spatial(imgs).index_select(1, idx).cpu())
        return torch.cat(out, dim=0)[np.argsort(order)]

    db = enc(db_indices, "[Proposed] DB")
    q = enc(target_indices - 1, "[Proposed] Q")
    mean_res, raw_res, _ = calculate_patch_mse_metrics(q.to(device), db.to(device), query_eval_dts, db_eval_dts)
    del model, db, q
    torch.cuda.empty_cache()
    return mean_res, raw_res


def main():
    d = run_dir(CONFIG)
    print(f"queries: {len(target_indices)} | DB: {len(db_indices)} | run_dir: {d}")

    for region in REGIONS_TO_RUN:
        raws, means = {}, {}

        print(f"\n=== [{region}] Proposed (no crop, patch selection) ===")
        means["Proposed"], raws["Proposed"] = eval_proposed(region)

        for bb in BACKBONE_SPECS.keys():
            print(f"\n=== [{region}] {bb}: G3 (whole-trained -> region crop) ===")
            m3, r3 = eval_backbone(bb, TRAINED_ON, region)
            if m3 is not None:
                means[f"{bb} (G3: whole-trained, crop@query)"] = m3
                raws[f"{bb} (G3: whole-trained, crop@query)"] = r3

            for pm in PAD_MODES:
                print(f"\n=== [{region}] {bb}: G3-pad ({pm}, scale preserved) ===")
                mp, rp = eval_backbone_pad(bb, TRAINED_ON, region, pm)
                if mp is not None:
                    means[f"{bb} (G3-pad/{pm}: whole-trained, crop@query)"] = mp
                    raws[f"{bb} (G3-pad/{pm}: whole-trained, crop@query)"] = rp

            print(f"\n=== [{region}] {bb}: G2 (trained on the same region, control) ===")
            m2, r2 = eval_backbone(bb, region, region)
            if m2 is not None:
                means[f"{bb} (G2: region-trained)"] = m2
                raws[f"{bb} (G2: region-trained)"] = r2

        # ---- mean table ----
        df = pd.DataFrame(means).T
        df.columns = [f"Top-{k}" for k in df.columns]
        df.to_csv(os.path.join(d, f"group3_wholetrain_crop_{region}.csv"))
        print(f"\n[{region}] mean weather-MSE\n{df.to_string()}")

        # ---- paired tests vs. Proposed + per-query raw values ----
        rows = []
        base = raws["Proposed"]
        for name, per_k in raws.items():
            for k, vals in per_k.items():
                v = np.asarray(vals, dtype=np.float64)
                if v.size == 0:
                    continue
                mean_v, lo, hi = bootstrap_ci(v)
                row = {"Model": name, "Top-k": k, "Mean MSE": mean_v,
                       "95% CI Low": lo, "95% CI High": hi,
                       "Median": float(np.median(v)), "N": len(v)}
                if name != "Proposed" and k in base:
                    b = np.asarray(base[k], dtype=np.float64)
                    if len(b) == len(v):
                        try:
                            row["Wilcoxon p (vs. Proposed)"] = wilcoxon(v, b).pvalue
                        except ValueError:
                            row["Wilcoxon p (vs. Proposed)"] = np.nan
                rows.append(row)
        sig = pd.DataFrame(rows)
        sig["Significant (p<0.05)"] = sig.get("Wilcoxon p (vs. Proposed)", np.nan) < 0.05
        sig.to_csv(os.path.join(d, f"group3_significance_{region}.csv"), index=False)
        print(f"\n[{region}] Top-1 summary")
        print(sig[sig["Top-k"] == 1].to_string(index=False))

        # keep per-query raw values for post hoc paired tests
        np.savez_compressed(
            os.path.join(d, f"group3_perquery_raw_{region}.npz"),
            **{f"{n}||Top-{k}": np.asarray(v, dtype=np.float64)
               for n, pk in raws.items() for k, v in pk.items()},
        )
        print(f"saved raws: group3_perquery_raw_{region}.npz")

    # ---- sanity check: on total_patches the result must equal G2 exactly ----
    print(f"\n=== sanity check (ckpt={TRAINED_ON}, eval={SELFCHECK_REGION}) ===")
    ref_path = os.path.join(d, f"group2_finetuned_{SELFCHECK_REGION}.csv")
    ref = pd.read_csv(ref_path, index_col=0) if os.path.exists(ref_path) else None
    check = {}
    for bb in BACKBONE_SPECS.keys():
        m, _ = eval_backbone(bb, TRAINED_ON, SELFCHECK_REGION)
        if m is None:
            continue
        check[bb] = m
        if ref is not None and bb in ref.index:
            got, exp = m[1], float(ref.loc[bb, "Top-1"] if "Top-1" in ref.columns else ref.loc[bb].iloc[0])
            ok = abs(got - exp) < 1e-9
            print(f"  {bb:10s} Top-1 got={got:.10f} expected={exp:.10f} -> {'MATCH' if ok else 'MISMATCH'}")
    if check:
        pd.DataFrame(check).T.to_csv(os.path.join(d, f"group3_selfcheck_{SELFCHECK_REGION}.csv"))
    print("\ndone.")


if __name__ == "__main__":
    main()
