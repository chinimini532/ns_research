"""
src/models/dtln_baseline.py

DTLN (Dual-Transform Learning Network) — External Baseline 2.

Original paper:
    Westhausen & Meyer (2020). Dual-Signal Transformation LSTM Network
    for Real-Time Noise Suppression. INTERSPEECH 2020.

This is the original DTLN architecture with LSTM — no modifications.
Used as both:
    - dtln_baseline: DTLN + MSE loss (baseline)
    - dtln_alaw:     DTLN + A-law loss (ablation — same architecture)

Input:  [batch, 160]  — one 20ms A-law decoded frame
Output: [batch, 160]  — denoised frame
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.config import N_BINS, DTLN_HIDDEN, DTLN_DROPOUT
from src.utils.audio import stft_torch, istft_torch, FRAME_SIZE


class DTLNBlock(nn.Module):
    """
    One DTLN processing block.

    Structure:
        LayerNorm → Dense(expand) → LSTM → Dense(mask) → Sigmoid

    The LSTM carries temporal context between frames via its hidden state.
    This is the original DTLN formulation — replaced by BCSU in our model.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.norm    = nn.LayerNorm(input_size)
        self.dense_in = nn.Linear(input_size, hidden_size)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm    = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dense_out = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

        # Persistent hidden state for inference
        self.register_buffer('h', torch.zeros(1, 1, hidden_size))
        self.register_buffer('c', torch.zeros(1, 1, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_size]

        Returns:
            mask: [batch, input_size] values in (0, 1)
        """
        batch = x.shape[0]

        out = self.norm(x)
        out = self.relu(self.dense_in(out))
        out = self.dropout(out)

        # LSTM expects [batch, seq, features]
        out = out.unsqueeze(1)

        if batch == 1:
            out, (self.h, self.c) = self.lstm(out, (self.h, self.c))
        else:
            out, _ = self.lstm(out)

        out = out.squeeze(1)
        mask = self.sigmoid(self.dense_out(out))
        return mask

    def reset_state(self):
        self.h.zero_()
        self.c.zero_()


class DTLNBaseline(nn.Module):
    """
    Original DTLN architecture — two-stage dual transform.

    Stage 1: frequency domain (STFT magnitude masking)
    Stage 2: time domain (waveform masking)

    Both stages use LSTM for temporal modeling.
    This is the unmodified baseline — LSTM not replaced by BCSU.
    """

    def __init__(
        self,
        hidden_size: int = DTLN_HIDDEN,
        dropout: float   = DTLN_DROPOUT
    ):
        super().__init__()

        # Stage 1 — frequency domain
        self.block1 = DTLNBlock(N_BINS, hidden_size, dropout)

        # Stage 2 — time domain
        self.block2 = DTLNBlock(FRAME_SIZE, hidden_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 160] noisy input frame

        Returns:
            [batch, 160] denoised output frame
        """
        # ── Stage 1: frequency domain ──────────────────────────
        magnitude, phase = stft_torch(x)       # [batch, 129]

        mask1 = self.block1(magnitude)         # [batch, 129]
        clean_mag = mask1 * magnitude          # [batch, 129]

        # Reconstruct intermediate signal
        intermediate = istft_torch(clean_mag, phase)  # [batch, 160]

        # ── Stage 2: time domain ───────────────────────────────
        mask2 = self.block2(intermediate)      # [batch, 160]
        output = mask2 * intermediate          # [batch, 160]

        return output

    def reset_state(self):
        self.block1.reset_state()
        self.block2.reset_state()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DTLNBaseline()
    x = torch.randn(4, 160)
    y = model(x)
    print(f"DTLN Baseline")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {model.count_parameters():,}")
    assert y.shape == x.shape
    print("  OK")