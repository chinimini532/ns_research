"""
src/models/rnnoise_bcsu.py

RNNoise-BCSU: RNNoise with Bounded Causal State Unit replacing GRU layers.

Original RNNoise uses two GRU layers for temporal modeling.
This model replaces both GRUs with BCSU blocks, providing:
    1. Formally bounded causal state
    2. Reduced parameter count
    3. Predictable memory usage for embedded deployment

Used in ablation study to show BCSU generalizes across architectures:
    RNNoise (GRU):      → baseline
    RNNoise-BCSU:       → proposed contribution on Architecture 3

Paper claim:
    "BCSU consistently improves upon GRU/LSTM temporal modeling across
     multiple architectures (DTLN, Conv-TasNet, RNNoise) when deployed
     in G.711 telephony domain."

Input/Output: [batch, 160] — one 20ms frame at 8kHz
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.config import N_BINS, RNNOISE_HIDDEN, BCSU_STATE_DIM, BCSU_HIDDEN
from src.utils.audio import stft_torch, istft_torch, FRAME_SIZE


class BCSU(nn.Module):
    """
    Bounded Causal State Unit — Novel Contribution.
    See bcsu_tasnet.py for full documentation.
    """
    def __init__(self, input_size, state_dim=BCSU_STATE_DIM, hidden_size=BCSU_HIDDEN):
        super().__init__()
        self.input_size = input_size
        self.state_dim  = state_dim
        self.fc1 = nn.Linear(input_size + state_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(hidden_size, input_size + state_dim)
        self.register_buffer('state', torch.zeros(1, state_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        state = self.state if batch == 1 else torch.zeros(
            batch, self.state_dim, device=x.device
        )
        combined  = torch.cat([x, state], dim=-1)
        out       = self.fc2(self.relu(self.fc1(combined)))
        features  = out[:, :self.input_size]
        new_state = out[:, self.input_size:]
        if batch == 1:
            self.state = new_state.detach()
        return features

    def reset_state(self):
        self.state.zero_()

    def state_memory_bytes(self):
        return self.state_dim * 4


class RNNoiseBCSU(nn.Module):
    """
    RNNoise with BCSU replacing GRU temporal modeling.

    Original RNNoise architecture:
        STFT → log magnitude → GRU1 → GRU2 → Sigmoid mask → ISTFT

    This model:
        STFT → log magnitude → BCSU1 → BCSU2 → Sigmoid mask → ISTFT

    The replacement is direct — BCSU receives the same input dimensions
    as GRU and produces the same output dimensions. No other changes.
    """
    def __init__(self, hidden=RNNOISE_HIDDEN):
        super().__init__()
        self.hidden = hidden

        # Input normalization
        self.input_norm = nn.LayerNorm(N_BINS)

        # Input projection
        self.input_proj = nn.Linear(N_BINS, hidden)
        self.relu1 = nn.ReLU()

        # BCSU1 replaces GRU1
        self.bcsu1 = BCSU(input_size=hidden)

        # BCSU2 replaces GRU2
        self.bcsu2 = BCSU(input_size=hidden)

        # Output mask
        self.fc_out  = nn.Linear(hidden, N_BINS)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 160] noisy speech frames
        Returns:
            [batch, 160] enhanced speech frames
        """
        # STFT
        magnitude, phase = stft_torch(x)

        # Feature extraction
        features = self.input_norm(torch.log1p(magnitude))
        features = self.relu1(self.input_proj(features))

        # BCSU temporal processing (replaces GRU1 + GRU2)
        features = self.bcsu1(features)
        features = self.bcsu2(features)

        # Mask estimation
        mask = self.sigmoid(self.fc_out(features))

        # Apply mask and reconstruct
        clean_magnitude = mask * magnitude
        output = istft_torch(clean_magnitude, phase)

        return output

    def reset_state(self):
        self.bcsu1.reset_state()
        self.bcsu2.reset_state()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def state_memory_bytes(self):
        return self.bcsu1.state_memory_bytes() + self.bcsu2.state_memory_bytes()


if __name__ == "__main__":
    model = RNNoiseBCSU()
    x = torch.randn(4, 160)
    y = model(x)
    print(f"RNNoiseBCSU")
    print(f"  Input:      {x.shape}")
    print(f"  Output:     {y.shape}")
    print(f"  Params:     {model.count_parameters():,}")
    print(f"  State RAM:  {model.state_memory_bytes()} bytes")
    assert y.shape == x.shape
    print("  OK")