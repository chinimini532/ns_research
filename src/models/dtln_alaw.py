"""
src/models/dtln_alaw.py

DTLN + A-law loss — Ablation Model 1.

Architecture identical to DTLNBaseline (original LSTM).
Trained with A-law domain loss instead of MSE.

This ablation isolates the contribution of the A-law loss function
independent of the BCSU architecture change.

Ablation table logic:
    dtln_baseline  → LSTM + MSE    (neither contribution)
    dtln_alaw      → LSTM + A-law  (loss contribution only)
    dtln_bcsu      → BCSU + MSE    (architecture contribution only)
    dtln_proposed  → BCSU + A-law  (both contributions)
"""

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.dtln_baseline import DTLNBaseline
from src.training.config import DTLN_HIDDEN, DTLN_DROPOUT


class DTLNAlawLoss(DTLNBaseline):
    """
    DTLN with original LSTM architecture.
    Trained with A-law domain loss (see src/training/loss.py).

    Architecture identical to DTLNBaseline.
    Only the loss function differs — handled in train.py.
    """

    def __init__(
        self,
        hidden_size: int = DTLN_HIDDEN,
        dropout: float   = DTLN_DROPOUT
    ):
        super().__init__(hidden_size=hidden_size, dropout=dropout)
        self.model_name = "DTLNAlawLoss"


if __name__ == "__main__":
    model = DTLNAlawLoss()
    x = torch.randn(4, 160)
    y = model(x)
    print(f"DTLN + A-law loss (ablation)")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {model.count_parameters():,}")
    assert y.shape == x.shape
    print("  OK")