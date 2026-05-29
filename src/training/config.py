"""
src/training/config.py
Central configuration for noise suppression research.
"""

from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"
OUT_MODELS  = ROOT / "outputs" / "models"
OUT_STATS   = ROOT / "outputs" / "stats"
OUT_FIGS    = ROOT / "outputs" / "figures"

for d in [DATA_RAW, DATA_PROC, DATA_SPLITS, OUT_MODELS, OUT_STATS, OUT_FIGS]:
    d.mkdir(parents=True, exist_ok=True)

# Audio
SOURCE_SR  = 16000
TARGET_SR  = 8000
FRAME_SIZE = 160
HOP_SIZE   = 80
FFT_SIZE   = 256
N_BINS     = 129

# Noise mixing
SNR_MIN    = 5.0    # Changed from -5 to +5 — easier task, more learnable
SNR_MAX    = 20.0

# Dataset
FRACTION   = 0.02
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# Model
DTLN_HIDDEN    = 256
DTLN_DROPOUT   = 0.25
BCSU_STATE_DIM = 64
BCSU_HIDDEN    = 256
RNNOISE_HIDDEN = 96

# Training
BATCH_SIZE  = 512      # good balance for A100
EPOCHS      = 200
LR          = 1e-3
LR_PATIENCE = 5
LR_FACTOR   = 0.5
EARLY_STOP  = 15
GRAD_CLIP   = 5.0
SEED        = 42
ALAW_ALPHA  = 1.0

# Device
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = "xpu"
else:
    DEVICE = "cpu"

# Experiments
EXPERIMENTS = [
    {
        "name":        "rnnoise_baseline",
        "model":       "rnnoise",
        "loss":        "mse",
        "description": "RNNoise + MSE loss (external baseline)"
    },
    {
        "name":        "dtln_baseline",
        "model":       "dtln_baseline",
        "loss":        "mse",
        "description": "DTLN original + MSE loss (external baseline)"
    },
    {
        "name":        "dtln_alaw",
        "model":       "dtln_baseline",
        "loss":        "alaw",
        "description": "DTLN + A-law loss only (ablation)"
    },
    {
        "name":        "dtln_bcsu",
        "model":       "dtln_bcsu",
        "loss":        "mse",
        "description": "DTLN + BCSU + masking + MSE (ablation)"
    },
    {
        "name":        "dtln_proposed_mask",
        "model":       "dtln_bcsu",
        "loss":        "alaw",
        "description": "BCSU + masking + A-law loss"
    },
    {
        "name":        "dtln_proposed",
        "model":       "dtln_proposed",
        "loss":        "alaw",
        "description": "BCSU + direct mapping + A-law (proposed)"
    },
    {
        "name":        "dtln_proposed_mse",
        "model":       "dtln_proposed",
        "loss":        "mse",
        "description": "BCSU + direct mapping + MSE (ablation)"
    },
    {
    "name":        "conv_tasnet",
    "model":       "conv_tasnet",
    "loss":        "mse",
    "description": "Conv-TasNet baseline retrained on A-law data"
    },
    {
    "name":        "bcsu_tasnet",
    "model":       "bcsu_tasnet",
    "loss":        "alaw",
    "description": "BCSU-TasNet proposed on Architecture 2"
    },
    {
    "name":        "rnnoise_bcsu",
    "model":       "rnnoise_bcsu",
    "loss":        "alaw",
    "description": "RNNoise + BCSU proposed on Architecture 3"
    },
]

# Evaluation
PESQ_MODE = "nb"

if __name__ == "__main__":
    print("=" * 55)
    print("  NS Research — Configuration")
    print("=" * 55)
    print(f"  Device:      {DEVICE}")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  Epochs:      {EPOCHS} (early stop @ {EARLY_STOP})")
    print(f"  LR:          {LR}")
    print(f"  SNR range:   {SNR_MIN} to {SNR_MAX} dB")
    print(f"  Experiments: {len(EXPERIMENTS)}")
    for e in EXPERIMENTS:
        print(f"    {e['name']:<20} loss={e['loss']}")
    print("=" * 55)