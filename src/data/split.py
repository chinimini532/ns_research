"""
src/data/split.py

Splits preprocessed frame pairs into train/val/test sets.

Input:
    data/processed/X_noisy.npy   [N, 160]
    data/processed/y_clean.npy   [N, 160]

Output:
    data/splits/X_train.npy, y_train.npy
    data/splits/X_val.npy,   y_val.npy
    data/splits/X_test.npy,  y_test.npy

Split ratio: 80% train / 10% val / 10% test
Shuffled before splitting to avoid speaker ordering bias.
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.config import (
    DATA_PROC, DATA_SPLITS,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED
)


def main():
    print("=" * 55)
    print("  NS Research — Train/Val/Test Split")
    print("=" * 55)

    # Load
    x_path = DATA_PROC / "X_noisy.npy"
    y_path = DATA_PROC / "y_clean.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Preprocessed data not found.\n"
            "Run: python src/data/preprocess.py"
        )

    print("  Loading preprocessed data...")
    X = np.load(x_path)
    y = np.load(y_path)

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    assert len(X) == len(y), "X and y must have same number of frames"

    # Shuffle
    np.random.seed(SEED)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split indices
    n       = len(X)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val   = X[n_train: n_train + n_val]
    y_val   = y[n_train: n_train + n_val]

    X_test  = X[n_train + n_val:]
    y_test  = y[n_train + n_val:]

    # Save
    DATA_SPLITS.mkdir(parents=True, exist_ok=True)

    np.save(DATA_SPLITS / "X_train.npy", X_train)
    np.save(DATA_SPLITS / "y_train.npy", y_train)
    np.save(DATA_SPLITS / "X_val.npy",   X_val)
    np.save(DATA_SPLITS / "y_val.npy",   y_val)
    np.save(DATA_SPLITS / "X_test.npy",  X_test)
    np.save(DATA_SPLITS / "y_test.npy",  y_test)

    print("\n  Split complete:")
    print(f"    Train: {len(X_train):,} frames  ({TRAIN_RATIO*100:.0f}%)")
    print(f"    Val:   {len(X_val):,} frames  ({VAL_RATIO*100:.0f}%)")
    print(f"    Test:  {len(X_test):,} frames  ({TEST_RATIO*100:.0f}%)")
    print(f"    Total: {n:,} frames")
    size_mb = sum(
        arr.nbytes for arr in [X_train, y_train, X_val, y_val, X_test, y_test]
    ) / 1e6
    print(f"    Size:  {size_mb:.1f} MB")
    print(f"  Saved to: {DATA_SPLITS}")
    print("=" * 55)
    print("\n  Next: python src/training/train.py")


if __name__ == "__main__":
    main()