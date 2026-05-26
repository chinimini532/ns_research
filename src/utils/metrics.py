"""
src/utils/metrics.py

Evaluation metrics for speech enhancement.

PESQ — Perceptual Evaluation of Speech Quality
    Standard metric for telephony speech quality.
    Range: -0.5 (worst) to 4.5 (best)
    Mode: 'nb' (narrowband) for 8kHz — correct for your system
    DO NOT use 'wb' (wideband) — that is for 16kHz systems

STOI — Short-Time Objective Intelligibility
    Measures how intelligible the speech is.
    Range: 0 (unintelligible) to 1 (perfect intelligibility)

Both are computed on the test set after training.
Neither is used during training — only MSE/A-law loss is used for training.
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.config import TARGET_SR, PESQ_MODE

# Try to import pesq — falls back to torchmetrics if not available
try:
    from pesq import pesq as pesq_fn
    PESQ_BACKEND = "pesq"
except ImportError:
    try:
        from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
        PESQ_BACKEND = "torchmetrics"
    except ImportError:
        PESQ_BACKEND = None

from pystoi import stoi as stoi_fn


def compute_pesq(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sr: int = TARGET_SR
) -> float:
    """
    Compute PESQ score between clean reference and enhanced signal.

    Args:
        clean:    clean reference signal, float32
        enhanced: enhanced/denoised signal, float32
        sr:       sample rate (must be 8000 for narrowband)

    Returns:
        PESQ score (float) or None if computation fails
    """
    # PESQ requires minimum signal length
    min_len = min(len(clean), len(enhanced))
    if min_len < sr * 0.5:  # at least 0.5 seconds
        return None

    clean    = clean[:min_len].astype(np.float32)
    enhanced = enhanced[:min_len].astype(np.float32)

    # Normalize to avoid clipping issues
    max_val = max(np.max(np.abs(clean)), np.max(np.abs(enhanced)), 1e-8)
    clean    = clean / max_val
    enhanced = enhanced / max_val

    try:
        if PESQ_BACKEND == "pesq":
            score = pesq_fn(sr, clean, enhanced, PESQ_MODE)
        elif PESQ_BACKEND == "torchmetrics":
            import torch
            metric = PerceptualEvaluationSpeechQuality(sr, PESQ_MODE)
            score = metric(
                torch.from_numpy(enhanced).unsqueeze(0),
                torch.from_numpy(clean).unsqueeze(0)
            ).item()
        else:
            return None
        return float(score)
    except Exception:
        return None


def compute_stoi(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sr: int = TARGET_SR
) -> float:
    """
    Compute STOI intelligibility score.

    Args:
        clean:    clean reference signal
        enhanced: enhanced signal
        sr:       sample rate

    Returns:
        STOI score (float) in [0, 1] or None if fails
    """
    min_len = min(len(clean), len(enhanced))
    if min_len < sr * 0.5:
        return None

    clean    = clean[:min_len].astype(np.float32)
    enhanced = enhanced[:min_len].astype(np.float32)

    try:
        score = stoi_fn(clean, enhanced, sr, extended=False)
        return float(score)
    except Exception:
        return None


def compute_snr(
    clean: np.ndarray,
    enhanced: np.ndarray
) -> float:
    """
    Compute output SNR between clean and enhanced signal.
    Simple metric — useful for quick monitoring during training.

    Returns:
        SNR in dB (higher is better)
    """
    noise = clean - enhanced
    signal_power = np.mean(clean ** 2) + 1e-8
    noise_power  = np.mean(noise ** 2) + 1e-8
    return float(10 * np.log10(signal_power / noise_power))


def evaluate_batch(
    clean_frames:    np.ndarray,
    enhanced_frames: np.ndarray,
    sr: int = TARGET_SR
) -> dict:
    """
    Evaluate a batch of frames.

    Reconstructs full utterances from frames, then computes
    PESQ and STOI on the full signal (required minimum length).

    Args:
        clean_frames:    [N, 160] clean reference frames
        enhanced_frames: [N, 160] model output frames
        sr:              sample rate

    Returns:
        dict with pesq, stoi, snr scores (None if computation failed)
    """
    # Flatten frames to continuous signal
    clean    = clean_frames.flatten().astype(np.float32)
    enhanced = enhanced_frames.flatten().astype(np.float32)

    return {
        "pesq": compute_pesq(clean, enhanced, sr),
        "stoi": compute_stoi(clean, enhanced, sr),
        "snr":  compute_snr(clean, enhanced)
    }


def aggregate_scores(scores_list: list) -> dict:
    """
    Aggregate a list of score dicts into mean ± std.

    Args:
        scores_list: list of dicts from evaluate_batch

    Returns:
        dict with mean and std for each metric
    """
    result = {}
    for metric in ["pesq", "stoi", "snr"]:
        values = [s[metric] for s in scores_list if s.get(metric) is not None]
        if values:
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"]  = float(np.std(values))
            result[f"{metric}_n"]    = len(values)
        else:
            result[f"{metric}_mean"] = None
            result[f"{metric}_std"]  = None
            result[f"{metric}_n"]    = 0
    return result


if __name__ == "__main__":
    print("Validating metrics...")
    print(f"  PESQ backend: {PESQ_BACKEND or 'not available'}")

    # Generate test signals
    sr = TARGET_SR
    t  = np.linspace(0, 1, sr, dtype=np.float32)
    clean    = np.sin(2 * np.pi * 440 * t) * 0.3
    noisy    = clean + np.random.randn(sr).astype(np.float32) * 0.05
    enhanced = clean + np.random.randn(sr).astype(np.float32) * 0.01

    pesq_score = compute_pesq(clean, enhanced, sr)
    stoi_score = compute_stoi(clean, enhanced, sr)
    snr_score  = compute_snr(clean, enhanced)

    print(f"  PESQ (enhanced): {pesq_score}")
    print(f"  STOI (enhanced): {stoi_score:.4f}" if stoi_score else "  STOI: None")
    print(f"  SNR  (enhanced): {snr_score:.2f} dB")
    print("  Metrics ready")