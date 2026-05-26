"""
src/data/preprocess.py
Chunked version - processes and saves in batches to avoid OOM.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.config import (
    DATA_RAW, DATA_PROC, FRACTION,
    SOURCE_SR, TARGET_SR, FRAME_SIZE, HOP_SIZE,
    SNR_MIN, SNR_MAX, SEED
)
from src.utils.audio import (
    load_audio, resample_to_8k, normalize_audio,
    mix_at_snr, make_frames
)
from src.utils.alaw import alaw_roundtrip_numpy


def collect_speech_files(fraction: float) -> list:
    kaggle_path = Path("/kaggle/input/datasets/yesha1910/librispeech/LibriSpeech/train-clean-100")
    local_path  = DATA_RAW / "librispeech" / "LibriSpeech" / "train-clean-100"
    dev_path    = DATA_RAW / "librispeech" / "LibriSpeech" / "dev-clean"

    if kaggle_path.exists():
        speech_root = kaggle_path
        print(f"  Using Kaggle LibriSpeech")
    elif local_path.exists():
        speech_root = local_path
        print(f"  Using local train-clean-100")
    elif dev_path.exists():
        speech_root = dev_path
        print(f"  Using local dev-clean")
    else:
        raise FileNotFoundError(f"LibriSpeech not found.\nChecked:\n  {kaggle_path}\n  {local_path}")

    files = list(speech_root.rglob("*.flac"))
    if not files:
        raise FileNotFoundError(f"No .flac files in {speech_root}")

    random.seed(SEED)
    random.shuffle(files)

    if fraction < 1.0:
        keep = max(10, int(len(files) * fraction))
        files = files[:keep]

    print(f"  Speech files: {len(files)}")
    return files


def collect_noise_files(fraction: float) -> list:
    noise_files = []

    kaggle_musan    = Path("/kaggle/input/datasets/nhattruongdev/musan-noise/musan/noise")
    local_musan     = DATA_RAW / "musan"
    local_synthetic = DATA_RAW / "musan_synthetic"

    if kaggle_musan.exists():
        noise_files.extend(list(kaggle_musan.rglob("*.wav")))
        print(f"  Using Kaggle MUSAN noise")
    elif local_musan.exists():
        noise_files.extend(list(local_musan.rglob("*.wav")))
        print(f"  Using local MUSAN")
    elif local_synthetic.exists():
        noise_files.extend(list(local_synthetic.rglob("*.wav")))
        print(f"  Using synthetic noise")
    else:
        raise FileNotFoundError("No noise files found.")

    if not noise_files:
        raise FileNotFoundError("No .wav files found in noise directory.")

    random.seed(SEED + 1)
    random.shuffle(noise_files)
    print(f"  Noise files: {len(noise_files)}")
    return noise_files


def load_and_resample(filepath: Path, source_sr: int) -> np.ndarray:
    try:
        audio = load_audio(str(filepath))
        audio = resample_to_8k(audio, source_sr)
        return audio
    except Exception:
        return None


def get_noise_segment(noise_files: list, length: int) -> np.ndarray:
    for _ in range(10):
        noise_path = random.choice(noise_files)
        try:
            noise = load_audio(str(noise_path))
        except Exception:
            continue
        if noise is None or len(noise) < 100:
            continue
        sr = TARGET_SR if "synthetic" in str(noise_path) else SOURCE_SR
        noise = resample_to_8k(noise, sr)
        if len(noise) < length:
            noise = np.tile(noise, (length // len(noise)) + 2)
        start = random.randint(0, len(noise) - length)
        return noise[start: start + length].astype(np.float32)
    return np.random.randn(length).astype(np.float32) * 0.1


def process_chunk(speech_files: list, noise_files: list) -> tuple:
    """Process a chunk of speech files and return frame arrays."""
    X_list = []
    y_list = []
    min_length = FRAME_SIZE * 4

    for speech_path in speech_files:
        speech = load_and_resample(speech_path, SOURCE_SR)
        if speech is None or len(speech) < min_length:
            continue

        speech = normalize_audio(speech, target_level=0.3)
        noise  = get_noise_segment(noise_files, len(speech))
        noise  = normalize_audio(noise, target_level=0.3)
        snr    = random.uniform(SNR_MIN, SNR_MAX)

        mixture      = mix_at_snr(speech, noise, snr_db=snr)
        noisy_input  = alaw_roundtrip_numpy(mixture)
        clean_target = speech

        noisy_frames = make_frames(noisy_input,  FRAME_SIZE, HOP_SIZE)
        clean_frames = make_frames(clean_target, FRAME_SIZE, HOP_SIZE)

        n_frames = min(len(noisy_frames), len(clean_frames))
        if n_frames == 0:
            continue

        X_list.append(noisy_frames[:n_frames])
        y_list.append(clean_frames[:n_frames])

    if not X_list:
        return None, None

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=FRACTION)
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Files per chunk (default 500). Reduce if OOM.")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 55)
    print("  NS Research — Preprocessing (Chunked)")
    print("=" * 55)
    print(f"  Fraction:   {args.fraction}")
    print(f"  Chunk size: {args.chunk_size} files per chunk")
    print(f"  Frame:      {FRAME_SIZE} samples = 20ms at {TARGET_SR}Hz")
    print(f"  SNR:        {SNR_MIN} to {SNR_MAX} dB")
    print("=" * 55)

    print("\nCollecting files...")
    speech_files = collect_speech_files(args.fraction)
    noise_files  = collect_noise_files(args.fraction)

    # Process in chunks and save to disk
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    x_path = DATA_PROC / "X_noisy.npy"
    y_path = DATA_PROC / "y_clean.npy"

    # Split speech files into chunks
    chunk_size  = args.chunk_size
    chunks      = [speech_files[i:i+chunk_size] for i in range(0, len(speech_files), chunk_size)]
    total_frames = 0
    all_X = []
    all_y = []

    print(f"\nProcessing {len(chunks)} chunks of ~{chunk_size} files each...")

    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i+1}/{len(chunks)} ({len(chunk)} files)...")
        X_chunk, y_chunk = process_chunk(chunk, noise_files)

        if X_chunk is None:
            print(f"  Chunk {i+1}: no valid frames, skipping")
            continue

        all_X.append(X_chunk)
        all_y.append(y_chunk)
        total_frames += len(X_chunk)
        print(f"  Chunk {i+1}: {len(X_chunk):,} frames | Total so far: {total_frames:,}")

        # Free chunk memory
        del X_chunk, y_chunk

    if not all_X:
        raise RuntimeError("No frames generated.")

    print("\nConcatenating and saving...")
    X_noisy = np.concatenate(all_X, axis=0).astype(np.float32)
    y_clean = np.concatenate(all_y, axis=0).astype(np.float32)

    np.save(x_path, X_noisy)
    np.save(y_path, y_clean)

    print("\n" + "=" * 55)
    print("  Preprocessing Complete")
    print("=" * 55)
    print(f"  X_noisy shape: {X_noisy.shape}")
    print(f"  y_clean shape: {y_clean.shape}")
    print(f"  Total frames:  {len(X_noisy):,}")
    print(f"  Total audio:   {len(X_noisy) * FRAME_SIZE / TARGET_SR / 60:.1f} minutes")
    size_mb = (X_noisy.nbytes + y_clean.nbytes) / 1e6
    print(f"  File size:     {size_mb:.1f} MB")
    print(f"  Saved to:      {DATA_PROC}")
    print("=" * 55)
    print("\n  Next: python src/data/split.py")


if __name__ == "__main__":
    main()