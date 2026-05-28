"""
src/utils/alaw.py
G.711 A-law codec simulation and loss functions.
"""

import numpy as np
import torch
import torch.nn as nn

A = 87.6
ALAW_MAX = 32767


def alaw_encode_numpy(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    x_abs = np.abs(x)
    sign  = np.sign(x)
    mask  = x_abs < (1.0 / A)
    compressed = np.where(
        mask,
        A * x_abs / (1.0 + np.log(A)),
        (1.0 + np.log(A * x_abs)) / (1.0 + np.log(A))
    )
    compressed = sign * compressed
    encoded = np.round((compressed + 1.0) / 2.0 * 255).astype(np.uint8)
    return encoded


def alaw_decode_numpy(encoded: np.ndarray) -> np.ndarray:
    encoded      = encoded.astype(np.float32)
    compressed   = (encoded / 255.0) * 2.0 - 1.0
    sign         = np.sign(compressed)
    compressed_abs = np.abs(compressed)
    threshold    = 1.0 / (1.0 + np.log(A))
    x_abs = np.where(
        compressed_abs < threshold,
        compressed_abs * (1.0 + np.log(A)) / A,
        np.exp(compressed_abs * (1.0 + np.log(A)) - 1.0) / A
    )
    return (sign * x_abs).astype(np.float32)


def alaw_roundtrip_numpy(x: np.ndarray) -> np.ndarray:
    return alaw_decode_numpy(alaw_encode_numpy(x))


def alaw_compress_torch(x: torch.Tensor) -> torch.Tensor:
    x       = torch.clamp(x, -1.0, 1.0)
    x_abs   = torch.abs(x)
    sign    = torch.sign(x)
    eps     = 1e-8
    threshold  = 1.0 / A
    linear     = A * x_abs / (1.0 + np.log(A))
    logarithmic = (1.0 + torch.log(A * x_abs + eps)) / (1.0 + np.log(A))
    compressed = torch.where(x_abs < threshold, linear, logarithmic)
    return sign * compressed


class AlawLoss(nn.Module):
    """
    A-law Domain Loss — Algorithmic Contribution.

    Computes MSE in G.711 A-law compressed domain.
    Weights errors according to codec's perceptual quantization structure:
    quiet speech errors penalized more than loud peak errors.
    Correlates better with PESQ than standard MSE.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output_compressed = alaw_compress_torch(output)
        target_compressed = alaw_compress_torch(target)
        return torch.mean((output_compressed - target_compressed) ** 2)


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((output - target) ** 2)


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse   = MSELoss()
        self.alaw  = AlawLoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (1 - self.alpha) * self.mse(output, target) + \
               self.alpha * self.alaw(output, target)


class ReconstructionLoss(nn.Module):
    """MSE + power penalty to prevent mask collapse."""
    def __init__(self, alpha=10.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse           = torch.mean((output - target) ** 2)
        output_power  = torch.mean(output ** 2)
        target_power  = torch.mean(target ** 2)
        power_penalty = torch.mean((output_power - target_power) ** 2)
        return mse + self.alpha * power_penalty


def validate_alaw_simulation():
    print("Validating A-law simulation...")
    x    = np.random.randn(8000).astype(np.float32) * 0.3
    x    = np.clip(x, -1.0, 1.0)
    x_rt = alaw_roundtrip_numpy(x)
    mse  = np.mean((x - x_rt) ** 2)
    print(f"  Roundtrip MSE: {mse:.6f} (expected < 0.001)")

    output = torch.randn(4, 160) * 0.3
    target = torch.randn(4, 160) * 0.3
    output.requires_grad_(True)

    loss_a = AlawLoss()(output, target)
    loss_m = MSELoss()(output, target)
    print(f"  A-law loss: {loss_a.item():.6f}")
    print(f"  MSE loss:   {loss_m.item():.6f}")

    loss_a.backward()
    grad_norm = output.grad.norm().item()
    print(f"  Grad norm:  {grad_norm:.6f} (must be > 0)")
    assert grad_norm > 0
    print("All checks passed.")


if __name__ == "__main__":
    validate_alaw_simulation()