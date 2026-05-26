"""
src/training/loss.py

Loss functions for noise suppression training.

Two losses:
    MSELoss   — standard mean squared error in linear PCM domain
    AlawLoss  — MSE computed in A-law compressed domain (our contribution)

The AlawLoss is imported from src/utils/alaw.py where the math
and paper justification are documented in detail.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Re-export from alaw.py for clean imports in train.py
from src.utils.alaw import AlawLoss, MSELoss, CombinedLoss

__all__ = ["AlawLoss", "MSELoss", "CombinedLoss"]