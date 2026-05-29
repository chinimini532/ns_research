"""
src/models/conv_tasnet.py

Conv-TasNet: Convolutional Time-domain Audio Separation Network
Baseline model for speech enhancement comparison.

Architecture:
    Learned encoder → TCN separator with dilated convolutions → Sigmoid mask → Learned decoder

Reference:
    Luo & Mesgarani (2019) "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking
    for Speech Separation" IEEE/ACM TASLP

Input/Output: [batch, 160] — one 20ms frame at 8kHz
Trained with: MSE loss on A-law distorted telephony data
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.audio import FRAME_SIZE


class TemporalBlock(nn.Module):
    """
    Single temporal convolutional block with dilated depthwise convolution.
    Uses residual connection for stable training.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.PReLU(),
            nn.GroupNorm(1, out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      dilation=dilation, padding=padding, groups=out_channels),
            nn.PReLU(),
            nn.GroupNorm(1, out_channels),
        )
        self.res = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.net(x) + self.res(x)


class ConvTasNet(nn.Module):
    """
    Conv-TasNet baseline for G.711 telephony speech enhancement.

    Trained from scratch on A-law distorted data — serves as strong
    deep learning baseline for comparison against BCSU-TasNet.

    Parameters:
        n_filters:  encoder filter count
        kernel_size: encoder/decoder kernel size
        stride:     encoder stride
        n_blocks:   temporal blocks per repeat
        n_repeats:  number of repeats
        hidden:     separator hidden size
    """
    def __init__(self,
                 n_filters=128,
                 kernel_size=16,
                 stride=8,
                 n_blocks=4,
                 n_repeats=2,
                 hidden=128):
        super().__init__()

        # Learned encoder (replaces STFT)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size, stride=stride, bias=False),
            nn.ReLU()
        )

        # TCN separator with exponential dilation
        separator_layers = []
        for r in range(n_repeats):
            for b in range(n_blocks):
                dilation = 2 ** b
                separator_layers.append(
                    TemporalBlock(n_filters, hidden, kernel_size=3, dilation=dilation)
                )
        self.separator = nn.Sequential(*separator_layers)

        # Sigmoid mask estimation
        self.mask = nn.Sequential(
            nn.Conv1d(hidden, n_filters, 1),
            nn.Sigmoid()
        )

        # Learned decoder
        self.decoder = nn.ConvTranspose1d(
            n_filters, 1, kernel_size, stride=stride, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 160] noisy speech frames
        Returns:
            [batch, 160] enhanced speech frames
        """
        x   = x.unsqueeze(1)           # [batch, 1, 160]
        enc = self.encoder(x)           # [batch, n_filters, T]
        sep = self.separator(enc)       # [batch, hidden, T]
        mask = self.mask(sep)           # [batch, n_filters, T]
        masked = enc * mask             # apply mask
        out = self.decoder(masked)      # [batch, 1, ~160]
        out = out[:, :, :FRAME_SIZE]    # trim to exactly 160
        return torch.tanh(out.squeeze(1))  # [batch, 160]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ConvTasNet()
    x = torch.randn(4, 160)
    y = model(x)
    print(f"ConvTasNet")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {model.count_parameters():,}")
    assert y.shape == x.shape
    print("  OK")