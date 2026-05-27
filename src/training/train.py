"""
src/training/train.py

Main training script. Trains all 5 models sequentially.

Usage:
    python src/training/train.py                        # train all models
    python src/training/train.py --model dtln_proposed  # train one model
"""

import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Fix path FIRST before any src imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.config import (
    DATA_SPLITS, DATA_PROC, OUT_MODELS, OUT_STATS,
    BATCH_SIZE, EPOCHS, LR, LR_PATIENCE, LR_FACTOR,
    EARLY_STOP, GRAD_CLIP, SEED, DEVICE, EXPERIMENTS
)
from src.training.loss import MSELoss, AlawLoss
from src.utils.metrics import evaluate_batch, aggregate_scores


# ─── Dataset ──────────────────────────────────────────────────────────────────

class NoisySpeechDataset(Dataset):
    def __init__(self, split: str):
        kaggle_path = Path("/kaggle/input/datasets/yesha1910/ns-research-splits/splits")

        if kaggle_path.exists():
            x_path = kaggle_path / f"X_{split}.npy"
            y_path = kaggle_path / f"y_{split}.npy"
        else:
            x_path = DATA_SPLITS / f"X_{split}.npy"
            y_path = DATA_SPLITS / f"y_{split}.npy"

        if not x_path.exists():
            raise FileNotFoundError(f"Split not found: {x_path}")

        print(f"  Loading {split} from {x_path.parent}...")
        self.X = torch.from_numpy(np.load(str(x_path))).float()
        self.y = torch.from_numpy(np.load(str(y_path))).float()
        print(f"  {split}: {len(self.X):,} frames")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Model Factory ────────────────────────────────────────────────────────────

def build_model(model_name: str) -> nn.Module:
    if model_name == "rnnoise":
        from src.models.rnnoise import RNNoise
        return RNNoise()
    elif model_name == "dtln_baseline":
        from src.models.dtln_baseline import DTLNBaseline
        return DTLNBaseline()
    elif model_name == "dtln_bcsu":
        from src.models.dtln_bcsu import DTLNWithBCSU
        return DTLNWithBCSU()
    elif model_name == "dtln_proposed":
        from src.models.dtln_proposed import DTLNProposed
        return DTLNProposed()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def build_loss(loss_name: str) -> nn.Module:
    if loss_name == "mse":
        return MSELoss()
    elif loss_name == "alaw":
        return AlawLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            output  = model(X_batch)
            loss    = criterion(output, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_test(model, loader, device) -> dict:
    model.eval()
    scores_list = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            output  = model(X_batch)
            enhanced_np = output.cpu().numpy()
            clean_np    = y_batch.numpy()
            scores = evaluate_batch(clean_np, enhanced_np)
            scores_list.append(scores)
    return aggregate_scores(scores_list)


# ─── Single Model Training ────────────────────────────────────────────────────

def train_model(experiment: dict, device: str) -> dict:
    name       = experiment["name"]
    model_type = experiment["model"]
    loss_type  = experiment["loss"]

    print(f"\n{'='*55}")
    print(f"  Training: {name}")
    print(f"  Model:    {model_type}  |  Loss: {loss_type}")
    print(f"{'='*55}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = NoisySpeechDataset("train")
    val_ds   = NoisySpeechDataset("val")
    test_ds  = NoisySpeechDataset("test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_ds):,} frames")
    print(f"  Val:   {len(val_ds):,} frames")
    print(f"  Test:  {len(test_ds):,} frames")

    model     = build_model(model_type).to(device)
    criterion = build_loss(loss_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_PATIENCE, factor=LR_FACTOR, verbose=False
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    best_val_loss    = float("inf")
    best_epoch       = 0
    epochs_no_improv = 0
    history          = {"train_loss": [], "val_loss": [], "lr": []}
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_STATS.mkdir(parents=True, exist_ok=True)
    save_path  = OUT_MODELS / f"{name}_best.pt"
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            epochs_no_improv = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improv += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS} | "
                f"train={train_loss:.4f} | "
                f"val={val_loss:.4f} | "
                f"best={best_val_loss:.4f} (ep{best_epoch}) | "
                f"lr={current_lr:.2e}"
            )

        if epochs_no_improv >= EARLY_STOP:
            print(f"  Early stopping at epoch {epoch}")
            break

    training_time = time.time() - start_time
    model.load_state_dict(torch.load(save_path, map_location=device))

    print(f"\n  Evaluating on test set...")
    test_scores = evaluate_test(model, test_loader, device)

    result = {
        "name":            name,
        "model":           model_type,
        "loss":            loss_type,
        "n_params":        n_params,
        "best_val_loss":   best_val_loss,
        "best_epoch":      best_epoch,
        "epochs_trained":  epoch,
        "training_time_s": round(training_time, 1),
        "save_path":       str(save_path),
        **test_scores
    }

    hist_path = OUT_STATS / f"{name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Results for {name}:")
    print(f"    PESQ:  {test_scores.get('pesq_mean')}")
    print(f"    STOI:  {test_scores.get('stoi_mean')}")
    print(f"    SNR:   {test_scores.get('snr_mean')}")
    print(f"    Time:  {training_time:.0f}s")

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = DEVICE
    print(f"\nDevice: {device}")

    if args.model:
        experiments = [e for e in EXPERIMENTS if e["name"] == args.model]
        if not experiments:
            print(f"Unknown model: {args.model}")
            print(f"Available: {[e['name'] for e in EXPERIMENTS]}")
            return
    else:
        experiments = EXPERIMENTS

    all_results = []
    total_start = time.time()

    for exp in experiments:
        result = train_model(exp, device)
        all_results.append(result)
        df = pd.DataFrame(all_results)
        df.to_csv(OUT_STATS / "all_training_stats.csv", index=False)

    total_time = time.time() - total_start

    print(f"\n{'='*55}")
    print(f"  All Training Complete")
    print(f"{'='*55}")
    print(f"  Total time: {total_time/60:.1f} minutes")

    df = pd.DataFrame(all_results)
    cols = ["name", "n_params", "best_val_loss", "pesq_mean", "stoi_mean", "snr_mean", "training_time_s"]
    available_cols = [c for c in cols if c in df.columns]
    print(df[available_cols].to_string(index=False))

    print(f"\n  Stats saved to: {OUT_STATS}")
    print(f"  Models saved to: {OUT_MODELS}")


if __name__ == "__main__":
    main()