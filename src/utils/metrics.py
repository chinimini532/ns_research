"""
src/utils/metrics.py

Evaluation metrics for speech enhancement.

PESQ — Perceptual Evaluation of Speech Quality
    Range: -0.5 (worst) to 4.5 (best)
    Mode: 'nb' (narrowband) for 8kHz telephony
    Requires minimum 0.5 seconds of audio

STOI — Short-Time Objective Intelligibility
    Range: 0 (unintelligible) to 1 (perfect)
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.config import TARGET_SR, PESQ_MODE

# PESQ backend detection
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


def compute_pesq(clean: np.ndarray, enhanced: np.ndarray, sr: int = TARGET_SR) -> float:
    """
    Compute PESQ score. Requires minimum 0.5 seconds of audio.
    Returns float score or None if computation fails.
    """
    min_len = min(len(clean), len(enhanced))
    min_samples = int(sr * 0.5)

    if min_len < min_samples:
        return None

    clean    = clean[:min_len].astype(np.float32)
    enhanced = enhanced[:min_len].astype(np.float32)

    # Normalize
    max_val = max(np.max(np.abs(clean)), np.max(np.abs(enhanced)), 1e-8)
    clean    = clean / max_val
    enhanced = enhanced / max_val

    try:
        if PESQ_BACKEND == "pesq":
            score = pesq_fn(sr, clean, enhanced, PESQ_MODE)
        elif PESQ_BACKEND == "torchmetrics":
            import torch
            metric = PerceptualEvaluationSpeechQuality(sr, PESQ_MODE)
            score  = metric(
                torch.from_numpy(enhanced).unsqueeze(0),
                torch.from_numpy(clean).unsqueeze(0)
            ).item()
        else:
            return None
        return float(score)
    except Exception:
        return None


def compute_stoi(clean: np.ndarray, enhanced: np.ndarray, sr: int = TARGET_SR) -> float:
    """Compute STOI intelligibility score. Returns float in [0,1] or None."""
    min_len = min(len(clean), len(enhanced))
    min_samples = int(sr * 0.5)

    if min_len < min_samples:
        return None

    clean    = clean[:min_len].astype(np.float32)
    enhanced = enhanced[:min_len].astype(np.float32)

    try:
        score = stoi_fn(clean, enhanced, sr, extended=False)
        return float(score)
    except Exception:
        return None


def compute_snr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """Compute output SNR in dB. Higher is better."""
    noise        = clean - enhanced
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

    Concatenates ALL frames into one continuous signal before
    computing PESQ and STOI — both require minimum signal length.

    Args:
        clean_frames:    [N, 160] clean reference frames
        enhanced_frames: [N, 160] model output frames

    Returns:
        dict with pesq, stoi, snr scores
    """
    # Flatten all frames into continuous signal
    clean    = clean_frames.flatten().astype(np.float32)
    enhanced = enhanced_frames.flatten().astype(np.float32)

    # Ensure minimum length for PESQ (0.5 seconds = 4000 samples at 8kHz)
    min_samples = int(sr * 0.5)
    if len(clean) < min_samples:
        repeats  = (min_samples // len(clean)) + 2
        clean    = np.tile(clean, repeats)[:min_samples]
        enhanced = np.tile(enhanced, repeats)[:min_samples]

    return {
        "pesq": compute_pesq(clean, enhanced, sr),
        "stoi": compute_stoi(clean, enhanced, sr),
        "snr":  compute_snr(clean, enhanced)
    }


def aggregate_scores(scores_list: list) -> dict:
    """
    Aggregate list of score dicts into mean and std.
    Filters out None values before computing statistics.
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
    print(f"PESQ backend: {PESQ_BACKEND or 'not available'}")

    sr   = TARGET_SR
    t    = np.linspace(0, 2, sr * 2, dtype=np.float32)
    clean    = np.sin(2 * np.pi * 440 * t) * 0.3
    enhanced = clean + np.random.randn(len(clean)).astype(np.float32) * 0.01

    pesq_score = compute_pesq(clean, enhanced, sr)
    stoi_score = compute_stoi(clean, enhanced, sr)
    snr_score  = compute_snr(clean, enhanced)

    print(f"PESQ: {pesq_score}")
    print(f"STOI: {stoi_score:.4f}" if stoi_score else "STOI: None")
    print(f"SNR:  {snr_score:.2f} dB")
    print("Metrics ready")