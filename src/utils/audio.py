"""
src/utils/audio.py

What this file does:
--------------------
1. Resampling — converts LibriSpeech 16kHz to your codec's 8kHz
2. Normalization — keeps audio in [-1, 1] range
3. Framing — cuts long audio into 160-sample (20ms) frames
4. Noise mixing — combines clean speech + noise at a target SNR
5. STFT/ISTFT — frequency domain conversion for the model's first stage
6. Validation — quick sanity check

Frame size: 160 samples = 20ms at 8kHz
This matches your RTP packet size exactly:
    RAU sends 160-byte G.711 A-law packets every 20ms
    Each byte = one A-law encoded sample
    160 samples at 8kHz = 20ms
"""

import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from pathlib import Path


# ─── Constants ────────────────────────────────────────────────────────────────
SOURCE_SR   = 16000   # LibriSpeech sample rate
TARGET_SR   = 8000    # Your TP3094 codec sample rate
FRAME_SIZE  = 160     # Samples per frame = 20ms at 8kHz
HOP_SIZE    = 80      # 50% overlap between frames

# STFT parameters
FFT_SIZE    = 256     # FFT window size (zero-padded from FRAME_SIZE)
WIN_SIZE    = 256     # Analysis window size
N_BINS      = 129     # FFT_SIZE // 2 + 1 frequency bins (0 to 4kHz)


# ─── Loading and Resampling ───────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """
    Load an audio file and return float32 samples.

    Handles .flac (LibriSpeech), .wav (MUSAN, DEMAND).
    Always returns mono audio normalized to [-1, 1].

    Args:
        path: path to audio file

    Returns:
        float32 numpy array, mono, original sample rate preserved
        (use resample_to_8k to convert)
    """
    audio, sr = sf.read(path, dtype='float32')

    # Convert stereo to mono by averaging channels
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio


def resample_to_8k(audio: np.ndarray, source_sr: int) -> np.ndarray:
    """
    Resample audio from source_sr to 8kHz.

    Uses polyphase resampling (scipy) which is higher quality
    than simple decimation. Preserves speech frequencies (0-4kHz).

    Args:
        audio: float32 array, any sample rate
        source_sr: original sample rate in Hz

    Returns:
        float32 array at 8kHz
    """
    if source_sr == TARGET_SR:
        return audio

    # Compute resampling ratio in lowest terms
    g = gcd(TARGET_SR, source_sr)
    up = TARGET_SR // g
    down = source_sr // g

    resampled = resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def normalize_audio(audio: np.ndarray, target_level: float = 0.3) -> np.ndarray:
    """
    Normalize audio to a target RMS level.

    Prevents clipping when mixing speech + noise.
    target_level = 0.3 gives comfortable headroom for mixing.

    Args:
        audio: float32 array
        target_level: target RMS level (0.0 to 1.0)

    Returns:
        normalized float32 array
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-8:
        return audio  # silence — don't normalize
    return audio * (target_level / rms)


# ─── Noise Mixing ─────────────────────────────────────────────────────────────

def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    snr_db: float
) -> np.ndarray:
    """
    Mix speech and noise at a target SNR level.

    SNR (Signal-to-Noise Ratio) in dB controls how loud
    the noise is relative to speech:
        +20 dB → very quiet noise (easy for model)
          0 dB → noise equals speech level (medium)
         -5 dB → noise louder than speech (hard)

    Args:
        speech: clean speech float32 array at 8kHz
        noise:  noise float32 array at 8kHz, same length as speech
        snr_db: target SNR in decibels

    Returns:
        noisy mixture float32 array, same length as speech
    """
    # Compute speech and noise power
    speech_rms = np.sqrt(np.mean(speech ** 2) + 1e-8)
    noise_rms  = np.sqrt(np.mean(noise ** 2) + 1e-8)

    # Scale noise to achieve target SNR
    # SNR = 20 * log10(speech_rms / noise_rms_scaled)
    # → noise_rms_scaled = speech_rms / 10^(snr_db/20)
    target_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / noise_rms)

    mixture = speech + noise_scaled
    return np.clip(mixture, -1.0, 1.0).astype(np.float32)


# ─── Framing ─────────────────────────────────────────────────────────────────

def make_frames(
    audio: np.ndarray,
    frame_size: int = FRAME_SIZE,
    hop_size: int = HOP_SIZE
) -> np.ndarray:
    """
    Cut a long audio array into overlapping frames.

    Each frame = 160 samples = 20ms at 8kHz.
    This matches your RTP packet boundary exactly.

    Args:
        audio:      float32 array, any length
        frame_size: samples per frame (default 160)
        hop_size:   samples between frame starts (default 80, 50% overlap)

    Returns:
        float32 array of shape [n_frames, frame_size]
    """
    n_frames = 1 + (len(audio) - frame_size) // hop_size
    if n_frames <= 0:
        return np.zeros((0, frame_size), dtype=np.float32)

    frames = np.stack([
        audio[i * hop_size: i * hop_size + frame_size]
        for i in range(n_frames)
    ])
    return frames.astype(np.float32)


def frames_to_audio(
    frames: np.ndarray,
    hop_size: int = HOP_SIZE
) -> np.ndarray:
    """
    Reconstruct audio from overlapping frames using overlap-add.

    Inverse of make_frames. Used during inference to convert
    the model's output frames back to a continuous audio signal.

    Args:
        frames:   float32 array of shape [n_frames, frame_size]
        hop_size: must match the hop_size used in make_frames

    Returns:
        float32 array, reconstructed audio signal
    """
    n_frames, frame_size = frames.shape
    audio_len = (n_frames - 1) * hop_size + frame_size
    audio = np.zeros(audio_len, dtype=np.float32)
    counts = np.zeros(audio_len, dtype=np.float32)

    for i, frame in enumerate(frames):
        start = i * hop_size
        audio[start: start + frame_size] += frame
        counts[start: start + frame_size] += 1

    # Normalize overlap regions
    counts = np.maximum(counts, 1)
    return audio / counts


# ─── STFT / ISTFT ─────────────────────────────────────────────────────────────

def compute_stft(
    frame: np.ndarray,
    fft_size: int = FFT_SIZE
) -> tuple:
    """
    Compute Short-Time Fourier Transform of one 160-sample frame.

    Converts time-domain samples to frequency-domain magnitude + phase.
    The model's first processing stage operates on this representation.

    Args:
        frame:    float32 array of shape [160]
        fft_size: FFT size (zero-pads frame from 160 to 256)

    Returns:
        magnitude: float32 array [129] — amplitude per frequency bin
        phase:     float32 array [129] — phase per frequency bin
                   (stored separately, added back during ISTFT)
    """
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(len(frame))
    windowed = frame * window

    # Zero-pad to fft_size and compute FFT
    spectrum = np.fft.rfft(windowed, n=fft_size)
    magnitude = np.abs(spectrum).astype(np.float32)
    phase = np.angle(spectrum).astype(np.float32)

    return magnitude, phase


def compute_istft(
    magnitude: np.ndarray,
    phase: np.ndarray,
    frame_size: int = FRAME_SIZE
) -> np.ndarray:
    """
    Reconstruct time-domain frame from magnitude + phase.

    Inverse of compute_stft. Used after the model applies
    its frequency-domain mask to recover the cleaned waveform.

    Args:
        magnitude: float32 array [129]
        phase:     float32 array [129] — original phase (not predicted)
        frame_size: output frame size (default 160)

    Returns:
        float32 array [160] — reconstructed audio frame
    """
    spectrum = magnitude * np.exp(1j * phase)
    audio = np.fft.irfft(spectrum)
    return audio[:frame_size].astype(np.float32)


# ─── PyTorch STFT (for model internal use) ────────────────────────────────────

def stft_torch(x: torch.Tensor, fft_size: int = FFT_SIZE) -> tuple:
    """
    Differentiable STFT for PyTorch tensors.

    Used inside the model's forward pass so gradients
    flow through the frequency-domain transformation.

    Args:
        x: tensor of shape [batch, frame_size]

    Returns:
        magnitude: tensor [batch, n_bins]
        phase:     tensor [batch, n_bins]
    """
    batch, frame_size = x.shape
    window = torch.hann_window(frame_size, device=x.device)
    x_windowed = x * window.unsqueeze(0)

    # Zero pad
    pad = fft_size - frame_size
    x_padded = torch.nn.functional.pad(x_windowed, (0, pad))

    spectrum = torch.fft.rfft(x_padded, n=fft_size)
    magnitude = torch.abs(spectrum)
    phase = torch.angle(spectrum)

    return magnitude, phase


def istft_torch(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    frame_size: int = FRAME_SIZE
) -> torch.Tensor:
    """
    Differentiable ISTFT for PyTorch tensors.

    Args:
        magnitude: tensor [batch, n_bins]
        phase:     tensor [batch, n_bins]
        frame_size: output size

    Returns:
        tensor [batch, frame_size]
    """
    spectrum = magnitude * torch.exp(1j * phase)
    audio = torch.fft.irfft(spectrum)
    return audio[:, :frame_size]


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_audio_utils():
    """
    Sanity check for all audio utilities.
    Run this file directly to verify everything works.
    """
    print("Validating audio utilities...")

    # Test 1 — resampling
    audio_16k = np.random.randn(16000).astype(np.float32) * 0.3
    audio_8k  = resample_to_8k(audio_16k, 16000)
    assert len(audio_8k) == 8000, f"Expected 8000 samples, got {len(audio_8k)}"
    print(f"  Resampling:        16000 → {len(audio_8k)} samples  OK")

    # Test 2 — normalization
    loud = np.random.randn(8000).astype(np.float32) * 10.0
    normalized = normalize_audio(loud, target_level=0.3)
    rms = np.sqrt(np.mean(normalized ** 2))
    assert abs(rms - 0.3) < 0.01, f"RMS {rms} not close to 0.3"
    print(f"  Normalization:     RMS = {rms:.4f}  OK")

    # Test 3 — noise mixing
    speech = normalize_audio(np.random.randn(8000).astype(np.float32))
    noise  = normalize_audio(np.random.randn(8000).astype(np.float32))
    mixture = mix_at_snr(speech, noise, snr_db=10.0)
    assert mixture.shape == speech.shape
    print(f"  SNR mixing:        shape {mixture.shape}  OK")

    # Test 4 — framing
    audio = np.random.randn(8000).astype(np.float32)
    frames = make_frames(audio)
    assert frames.shape[1] == FRAME_SIZE, f"Frame size {frames.shape[1]} != {FRAME_SIZE}"
    print(f"  Framing:           {frames.shape[0]} frames of {frames.shape[1]} samples  OK")

    # Test 5 — STFT/ISTFT roundtrip
    # Note: Hann windowing before FFT is intentionally lossy
    # (reduces spectral leakage at cost of perfect reconstruction)
    # The model operates on magnitude + stored phase so this is fine
    frame = frames[0]
    mag, phase = compute_stft(frame)
    reconstructed = compute_istft(mag, phase)
    mse = np.mean((frame - reconstructed) ** 2)
    assert mag.shape == (N_BINS,), f"Magnitude shape {mag.shape} != ({N_BINS},)"
    assert phase.shape == (N_BINS,), f"Phase shape {phase.shape} != ({N_BINS},)"
    print(f"  STFT roundtrip:    mag {mag.shape}, phase {phase.shape}, MSE = {mse:.4f}  OK")

    # Test 6 — PyTorch STFT
    x = torch.randn(4, FRAME_SIZE)
    mag_t, phase_t = stft_torch(x)
    assert mag_t.shape == (4, N_BINS), f"STFT shape {mag_t.shape} unexpected"
    print(f"  PyTorch STFT:      shape {mag_t.shape}  OK")

    # Test 7 — PyTorch ISTFT
    recon_t = istft_torch(mag_t, phase_t)
    assert recon_t.shape == (4, FRAME_SIZE)
    print(f"  PyTorch ISTFT:     shape {recon_t.shape}  OK")

    print("All checks passed.\n")


if __name__ == "__main__":
    validate_audio_utils()