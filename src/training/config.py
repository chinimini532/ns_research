"""
src/training/config.py

Central configuration for the noise suppression project.
Change values here only — everything else reads from this file.

Experiment matrix (5 models, all trained on same data):
    1. rnnoise_baseline    → RNNoise architecture + MSE loss
    2. dtln_baseline       → DTLN original      + MSE loss
    3. dtln_alaw           → DTLN original      + A-law loss   (ablation)
    4. dtln_bcsu           → DTLN + BCSU        + MSE loss     (ablation)
    5. dtln_proposed       → DTLN + BCSU        + A-law loss   (proposed)
"""

from pathlib import Path

# ─── Root Paths ───────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"
OUT_MODELS  = ROOT / "outputs" / "models"
OUT_STATS   = ROOT / "outputs" / "stats"
OUT_FIGS    = ROOT / "outputs" / "figures"

# Create directories if they don't exist
for d in [DATA_RAW, DATA_PROC, DATA_SPLITS, OUT_MODELS, OUT_STATS, OUT_FIGS]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Audio ────────────────────────────────────────────────────────────────────
SOURCE_SR   = 16000   # LibriSpeech sample rate
TARGET_SR   = 8000    # TP3094 codec sample rate
FRAME_SIZE  = 160     # Samples per frame = 20ms at 8kHz
HOP_SIZE    = 80      # 50% overlap
FFT_SIZE    = 256     # STFT window (zero-padded)
N_BINS      = 129     # FFT_SIZE // 2 + 1

# ─── Noise Mixing ─────────────────────────────────────────────────────────────
SNR_MIN     = -5.0    # dB — hardest condition (noise louder than speech)
SNR_MAX     = 20.0    # dB — easiest condition (speech clearly audible)

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Fraction of dataset to use
# 0.02 = LG Gram development (fast, confirms code works)
# 1.00 = Kaggle full training (publication quality results)
FRACTION    = 0.02

TRAIN_RATIO = 0.80    # 80% training
VAL_RATIO   = 0.10    # 10% validation
TEST_RATIO  = 0.10    # 10% test

# ─── Model Architecture ───────────────────────────────────────────────────────
# DTLN shared parameters
DTLN_HIDDEN     = 256     # Hidden units in Dense layers
DTLN_DROPOUT    = 0.25    # Dropout rate

# BCSU parameters (your novel component)
BCSU_STATE_DIM  = 64      # Fixed state vector size (bytes = 64 * 4 = 256 bytes)
                           # This is provably bounded — important for paper
BCSU_HIDDEN     = 256     # Hidden units inside BCSU feedforward network

# RNNoise parameters
RNNOISE_HIDDEN  = 96      # RNNoise uses smaller hidden size by design

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32      # Frames per batch
EPOCHS          = 100     # Maximum epochs
LR              = 1e-3    # Initial learning rate (Adam)
LR_PATIENCE     = 5       # Epochs before LR reduction
LR_FACTOR       = 0.5     # LR multiplied by this on plateau
EARLY_STOP      = 15      # Epochs without improvement before stopping
GRAD_CLIP       = 5.0     # Gradient clipping (important for BCSU stability)
SEED            = 42      # Reproducibility

# A-law combined loss alpha
# 0.0 = pure MSE, 1.0 = pure A-law, 0.5 = equal mix
ALAW_ALPHA      = 1.0     # Use pure A-law loss for proposed model

# ─── Device ───────────────────────────────────────────────────────────────────
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = "xpu"
else:
    DEVICE = "cpu"

# ─── Experiment Definitions ───────────────────────────────────────────────────
# Each experiment is a dict with model name and loss type
# Training script iterates over all experiments automatically

EXPERIMENTS = [
    {
        "name":       "rnnoise_baseline",
        "model":      "rnnoise",
        "loss":       "mse",
        "description": "RNNoise architecture with standard MSE loss (external baseline)"
    },
    {
        "name":       "dtln_baseline",
        "model":      "dtln_baseline",
        "loss":       "mse",
        "description": "DTLN original architecture with standard MSE loss (external baseline)"
    },
    {
        "name":       "dtln_alaw",
        "model":      "dtln_baseline",
        "loss":       "alaw",
        "description": "DTLN original with A-law domain loss only (ablation)"
    },
    {
        "name":       "dtln_bcsu",
        "model":      "dtln_bcsu",
        "loss":       "mse",
        "description": "DTLN with BCSU architecture, standard MSE loss (ablation)"
    },
    {
        "name":       "dtln_proposed",
        "model":      "dtln_proposed",
        "loss":       "alaw",
        "description": "Proposed: DTLN + BCSU + A-law domain loss (full contribution)"
    },
]

# ─── Evaluation ───────────────────────────────────────────────────────────────
# PESQ mode: 'nb' = narrowband (correct for 8kHz telephony)
# 'wb' = wideband (for 16kHz — do NOT use for your system)
PESQ_MODE   = "nb"

# ─── Quick Sanity Check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  NS Research — Configuration")
    print("=" * 55)
    print(f"  Device:       {DEVICE}")
    print(f"  Fraction:     {FRACTION}")
    print(f"  Frame size:   {FRAME_SIZE} samples = 20ms at {TARGET_SR}Hz")
    print(f"  Batch size:   {BATCH_SIZE}")
    print(f"  Epochs:       {EPOCHS} (early stop @ {EARLY_STOP})")
    print(f"  LR:           {LR}")
    print(f"  BCSU state:   {BCSU_STATE_DIM} dims = {BCSU_STATE_DIM * 4} bytes RAM")
    print(f"  SNR range:    {SNR_MIN} to {SNR_MAX} dB")
    print()
    print(f"  Experiments ({len(EXPERIMENTS)} total):")
    for i, exp in enumerate(EXPERIMENTS):
        print(f"    {i+1}. {exp['name']:<20} loss={exp['loss']}")
    print()
    print(f"  Paths:")
    print(f"    Raw data:   {DATA_RAW}")
    print(f"    Processed:  {DATA_PROC}")
    print(f"    Models:     {OUT_MODELS}")
    print(f"    Stats:      {OUT_STATS}")
    print("=" * 55)