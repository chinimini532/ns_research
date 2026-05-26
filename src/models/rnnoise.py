"""
src/models/rnnoise.py

RNNoise-inspired architecture — External Baseline 1.

Original RNNoise (Mozilla, 2018) uses GRU layers to suppress noise
in the frequency domain. This is a faithful reimplementation of
the core architecture adapted for 8kHz telephony input.

Paper reference:
    Valin, J.M. (2018). A Hybrid DSP/Deep Learning Approach to
    Real-Time Full-Band Speech Enhancement. MMSP 2018.

Input:  [batch, 160]  — one 20ms A-law decoded frame
Output: [batch, 160]  — denoised frame
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.config import N_BINS, RNNOISE_HIDDEN
from src.utils.audio import stft_torch, istft_torch


class RNNoise(nn.Module):
    """
    RNNoise-inspired speech enhancement model.

    Architecture:
        STFT → GRU (frequency domain) → mask → ISTFT

    The GRU processes frequency features across time and outputs
    a suppression mask per frequency bin. The mask is applied to
    the magnitude spectrum before ISTFT reconstruction.

    This is an external baseline — not modified for our contribution.
    Trained with standard MSE loss.
    """

    def __init__(self, hidden_size: int = RNNOISE_HIDDEN):
        super().__init__()
        self.hidden_size = hidden_size

        # Input: log magnitude spectrum [batch, N_BINS=129]
        self.input_norm = nn.LayerNorm(N_BINS)

        # GRU processes spectral features
        self.gru1 = nn.GRU(
            input_size=N_BINS,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False  # causal — no future lookahead
        )

        self.gru2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False
        )

        # Output: suppression mask per frequency bin
        self.fc_out = nn.Linear(hidden_size, N_BINS)
        self.sigmoid = nn.Sigmoid()

        # Hidden state — maintained across frames during inference
        self.register_buffer('h1', torch.zeros(1, 1, hidden_size))
        self.register_buffer('h2', torch.zeros(1, 1, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 160] noisy input frame

        Returns:
            [batch, 160] denoised output frame
        """
        batch = x.shape[0]

        # STFT
        magnitude, phase = stft_torch(x)          # [batch, 129]

        # Log compression for better numerical range
        log_mag = torch.log1p(magnitude)           # [batch, 129]

        # Normalize
        log_mag = self.input_norm(log_mag)

        # GRU expects [batch, seq_len, features]
        # Here seq_len=1 (single frame, stateful processing)
        features = log_mag.unsqueeze(1)            # [batch, 1, 129]

        # GRU layers
        if batch == 1:
            # Inference: use and update persistent hidden state
            out1, self.h1 = self.gru1(features, self.h1)
            out2, self.h2 = self.gru2(out1, self.h2)
        else:
            # Training: fresh hidden state per batch
            out1, _ = self.gru1(features)
            out2, _ = self.gru2(out1)

        out2 = out2.squeeze(1)                     # [batch, hidden]

        # Suppression mask [0, 1] per frequency bin
        mask = self.sigmoid(self.fc_out(out2))     # [batch, 129]

        # Apply mask to original magnitude
        clean_magnitude = mask * magnitude         # [batch, 129]

        # ISTFT reconstruction using original phase
        output = istft_torch(clean_magnitude, phase)  # [batch, 160]

        return output

    def reset_state(self):
        """Reset hidden state between utterances during inference."""
        self.h1.zero_()
        self.h2.zero_()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = RNNoise()
    x = torch.randn(4, 160)
    y = model(x)
    print(f"RNNoise")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {model.count_parameters():,}")
    assert y.shape == x.shape, "Output shape mismatch"
    print("  OK")