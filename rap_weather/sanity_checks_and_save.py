"""Post-training sanity checks, artifact saving, and GPU cleanup."""
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CONFIG, device, embedding_save_path, model_save_path, run_dir
from .models import cleanup_gpu


def debug_model_output(model, val_loader):
    print("\nDebugging Model Outputs...")
    model.eval()
    seq, target, idx = next(iter(val_loader))
    seq, target = seq.to(device), target.to(device)
    with torch.no_grad():
        pred_emb = model(seq)
        gt_emb = model.forward_spatial(target)
        print(f"Pred Sample (First 5): {pred_emb[0, 0, :5].cpu().numpy()}")
        print(f"GT   Sample (First 5): {gt_emb[0, 0, :5].cpu().numpy()}")
        print(f"Pred Stats | Min: {pred_emb.min():.4f}, Max: {pred_emb.max():.4f}, Mean: {pred_emb.mean():.4f}, Std: {pred_emb.std():.4f}")
        if pred_emb.std() < 1e-6:
            print("CRITICAL: Prediction Vector Collapsed (All values are same)!")
        pred_norm = F.normalize(pred_emb, p=2, dim=2)
        gt_norm = F.normalize(gt_emb, p=2, dim=2)
        sim = F.cosine_similarity(pred_norm, gt_norm, dim=2)
        print(f"Cosine Sim (Mean/Max/Min): {sim.mean().item():.6f} / {sim.max().item():.6f} / {sim.min().item():.6f}")
        img_diff = F.mse_loss(seq[:, -1, :, :, :], target)
        print(f"Input(t) vs Target(t+1) Pixel MSE: {img_diff.item():.6f}")
        if img_diff.item() == 0:
            print("CRITICAL: Input and Target images are IDENTICAL! (Data Leakage)")
        gt_0, gt_1 = gt_emb[0].flatten(), gt_emb[1].flatten()
        print(f"Sample 0 vs Sample 1 Similarity: {F.cosine_similarity(gt_0, gt_1, dim=0).item():.6f}")


def save_artifacts(model, db_embeddings, db_datetimes, repro_config, cfg=CONFIG):
    save_dir = run_dir(cfg)
    print(f"Saving artifacts to {save_dir}...")
    torch.save(model.state_dict(), model_save_path(cfg))
    print(f"Model weights saved: {model_save_path(cfg)}")

    db_data = {
        "embeddings": db_embeddings.cpu(),
        "datetimes": db_datetimes,
        "model_config": repro_config,   # full reproducibility manifest
    }
    with open(embedding_save_path(cfg), "wb") as f:
        pickle.dump(db_data, f)
    print(f"DB Embeddings saved: {embedding_save_path(cfg)}")


def main():
    from . import train as train_stage

    (model, optimizer, scaler, train_ds, val_ds, train_dts, val_dts, train_imgs,
     db_embeddings, db_datetimes, repro_config) = train_stage.main()

    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=8)
    debug_model_output(model, val_loader)

    save_artifacts(model, db_embeddings, db_datetimes, repro_config)

    # --- GPU cleanup ---
    del model, optimizer, scaler, db_embeddings, val_loader
    cleanup_gpu("model", "optimizer", "scaler", "db_embeddings", "val_loader")


if __name__ == "__main__":
    main()
