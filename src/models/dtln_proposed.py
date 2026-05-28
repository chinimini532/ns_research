"""
src/models/dtln_proposed.py
Proposed model - BCSU + A-law loss + direct waveform mapping
No Sigmoid mask (prevents collapse), uses residual connections + Tanh output
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.config import DTLN_HIDDEN, DTLN_DROPOUT, BCSU_STATE_DIM, BCSU_HIDDEN
from src.utils.audio import FRAME_SIZE


class BCSU(nn.Module):
    """
    Bounded Causal State Unit — Novel Contribution.
    Replaces LSTM with explicit fixed-size state vector.
    State: exactly 64 dims = 256 bytes RAM, provably causal.
    """
    def __init__(self, input_size, state_dim=BCSU_STATE_DIM, hidden_size=BCSU_HIDDEN):
        super().__init__()
        self.input_size = input_size
        self.state_dim  = state_dim
        self.fc1 = nn.Linear(input_size + state_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(hidden_size, input_size + state_dim)
        self.register_buffer('state', torch.zeros(1, state_dim))

    def forward(self, x):
        batch = x.shape[0]
        state = self.state if batch == 1 else torch.zeros(
            batch, self.state_dim, device=x.device
        )
        combined = torch.cat([x, state], dim=-1)
        out      = self.fc2(self.relu(self.fc1(combined)))
        features  = out[:, :self.input_size]
        new_state = out[:, self.input_size:]
        if batch == 1:
            self.state = new_state.detach()
        return features

    def reset_state(self):
        self.state.zero_()

    def state_memory_bytes(self):
        return self.state_dim * 4


class DTLNProposed(nn.Module):
    """
    Proposed model: Two BCSU blocks with direct waveform mapping.

    Architecture:
        Input [batch, 160]
        → Block 1: LayerNorm → Dense → BCSU → Dense → Residual
        → Block 2: LayerNorm → Dense → BCSU → Dense → Residual
        → Tanh output [-1, 1]

    Key differences from baseline DTLN:
    1. BCSU replaces LSTM (bounded causal state, 60% fewer params)
    2. Direct mapping with Tanh instead of Sigmoid mask (no collapse)
    3. Residual connections for stable training
    4. Trained with A-law domain loss (codec-aware)
    """
    def __init__(self, hidden_size=DTLN_HIDDEN, dropout=DTLN_DROPOUT):
        super().__init__()

        # Block 1
        self.norm1      = nn.LayerNorm(FRAME_SIZE)
        self.dense1_in  = nn.Linear(FRAME_SIZE, hidden_size)
        self.relu1      = nn.ReLU()
        self.drop1      = nn.Dropout(dropout)
        self.bcsu1      = BCSU(input_size=hidden_size)
        self.dense1_out = nn.Linear(hidden_size, FRAME_SIZE)

        # Block 2
        self.norm2      = nn.LayerNorm(FRAME_SIZE)
        self.dense2_in  = nn.Linear(FRAME_SIZE, hidden_size)
        self.relu2      = nn.ReLU()
        self.drop2      = nn.Dropout(dropout)
        self.bcsu2      = BCSU(input_size=hidden_size)
        self.dense2_out = nn.Linear(hidden_size, FRAME_SIZE)

        self.tanh = nn.Tanh()

    def forward(self, x):
        # Block 1 with residual
        out  = self.norm1(x)
        out  = self.drop1(self.relu1(self.dense1_in(out)))
        out  = self.bcsu1(out)
        out  = x + self.dense1_out(out)

        # Block 2 with residual
        out2 = self.norm2(out)
        out2 = self.drop2(self.relu2(self.dense2_in(out2)))
        out2 = self.bcsu2(out2)
        out2 = out + self.dense2_out(out2)

        return self.tanh(out2)

    def reset_state(self):
        self.bcsu1.reset_state()
        self.bcsu2.reset_state()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def state_memory_bytes(self):
        return self.bcsu1.state_memory_bytes() + self.bcsu2.state_memory_bytes()


if __name__ == "__main__":
    model = DTLNProposed()
    x = torch.randn(4, 160)
    y = model(x)
    print(f"DTLNProposed")
    print(f"  Input:      {x.shape}")
    print(f"  Output:     {y.shape}")
    print(f"  Params:     {model.count_parameters():,}")
    print(f"  State RAM:  {model.state_memory_bytes()} bytes")
    assert y.shape == x.shape
    print("  OK")