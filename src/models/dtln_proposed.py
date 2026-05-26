"""
src/models/dtln_proposed.py

Proposed Model — DTLN + BCSU + A-law Domain Loss.

This is the full contribution combining both novelties:
    1. BCSU architecture (from dtln_bcsu.py)
    2. A-law domain loss (from src/training/loss.py)

The model architecture is identical to DTLNWithBCSU.
The difference is in training — this model is trained
with AlawLoss instead of MSELoss.

Having a separate file makes the experiment table clean
and makes it explicit which model is "proposed".
"""

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Proposed model reuses DTLNWithBCSU architecture exactly
# Only the loss function differs — handled in train.py
from src.models.dtln_bcsu import DTLNWithBCSU
from src.training.config import DTLN_HIDDEN, DTLN_DROPOUT


class DTLNProposed(DTLNWithBCSU):
    """
    Proposed model: DTLN + BCSU + A-law domain loss.

    Architecture identical to DTLNWithBCSU.
    Trained with AlawLoss (see src/training/loss.py).

    This class exists to:
    1. Make the experiment table explicit
    2. Allow model-specific hyperparameter tuning if needed
    3. Clearly label the proposed model in saved checkpoints
    """

    def __init__(
        self,
        hidden_size: int = DTLN_HIDDEN,
        dropout: float   = DTLN_DROPOUT
    ):
        super().__init__(hidden_size=hidden_size, dropout=dropout)
        self.model_name = "DTLNProposed"


if __name__ == "__main__":
    model = DTLNProposed()
    x = torch.randn(4, 160)
    y = model(x)
    print(f"DTLN Proposed (BCSU + A-law loss)")
    print(f"  Input:      {x.shape}")
    print(f"  Output:     {y.shape}")
    print(f"  Params:     {model.count_parameters():,}")
    print(f"  State RAM:  {model.state_memory_bytes()} bytes")
    assert y.shape == x.shape
    print("  OK")