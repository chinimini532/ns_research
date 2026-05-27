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

    # Kaggle mounted MUSAN noise
    kaggle_musan = Path("/kaggle/input/datasets/nhattruongdev/musan-noise/musan/noise")
    # Local ESC-50
    local_esc50  = DATA_RAW / "ESC-50-master" / "audio"
    # Local DEMAND
    local_demand = DATA_RAW / "Demand"
    # Local synthetic
    local_synth  = DATA_RAW / "musan_synthetic"

    if kaggle_musan.exists():
        noise_files.extend(list(kaggle_musan.rglob("*.wav")))
        print(f"  Using Kaggle MUSAN noise: {len(noise_files)} files")

    if local_esc50.exists():
        esc_files = list(local_esc50.glob("*.wav"))
        noise_files.extend(esc_files)
        print(f"  Using ESC-50: {len(esc_files)} files")

    if local_demand.exists():
        demand_files = list(local_demand.rglob("*.wav"))
        noise_files.extend(demand_files)
        print(f"  Using DEMAND: {len(demand_files)} files")

    if local_synth.exists():
        synth_files = list(local_synth.glob("*.wav"))
        noise_files.extend(synth_files)
        print(f"  Using synthetic: {len(synth_files)} files")

    if not noise_files:
        raise FileNotFoundError("No noise files found.")

    random.seed(SEED + 1)
    random.shuffle(noise_files)
    print(f"  Total noise files: {len(noise_files)}")
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

        # Determine sample rate from path
        path_str = str(noise_path)
        if "synthetic" in path_str:
            sr = TARGET_SR
        else:
            sr = SOURCE_SR  # ESC-50, DEMAND, MUSAN all 16kHz

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
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 55)
    print("  NS Research — Preprocessing")
    print("=" * 55)
    print(f"  Fraction:   {args.fraction}")
    print(f"  Chunk size: {args.chunk_size}")
    print("=" * 55)

    print("\nCollecting files...")
    speech_files = collect_speech_files(args.fraction)
    noise_files  = collect_noise_files(args.fraction)

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    chunk_dir = DATA_PROC / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    chunks = [
        speech_files[i:i+args.chunk_size]
        for i in range(0, len(speech_files), args.chunk_size)
    ]

    total_frames = 0

    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i+1}/{len(chunks)}...")
        X, y = process_chunk(chunk, noise_files)
        if X is None:
            print(f"  Chunk {i+1}: skipped")
            continue
        # Save immediately to disk
        np.save(chunk_dir / f"X_{i:04d}.npy", X)
        np.save(chunk_dir / f"y_{i:04d}.npy", y)
        total_frames += len(X)
        print(f"  Saved {len(X):,} | Total: {total_frames:,}")
        del X, y

    # Merge using memmap — no RAM needed
    print("\nMerging chunks to final files...")
    x_files = sorted(chunk_dir.glob("X_*.npy"))
    y_files = sorted(chunk_dir.glob("y_*.npy"))

    x_out = DATA_PROC / "X_noisy.npy"
    y_out = DATA_PROC / "y_clean.npy"

    X_mm = np.lib.format.open_memmap(
        str(x_out), mode='w+', dtype='float32',
        shape=(total_frames, FRAME_SIZE)
    )
    y_mm = np.lib.format.open_memmap(
        str(y_out), mode='w+', dtype='float32',
        shape=(total_frames, FRAME_SIZE)
    )

    offset = 0
    for xf, yf in zip(x_files, y_files):
        Xc = np.load(xf)
        yc = np.load(yf)
        n  = len(Xc)
        X_mm[offset:offset+n] = Xc
        y_mm[offset:offset+n] = yc
        offset += n
        del Xc, yc
        xf.unlink()
        yf.unlink()

    del X_mm, y_mm
    chunk_dir.rmdir()

    print("\n" + "=" * 55)
    print("  Preprocessing Complete")
    print("=" * 55)
    print(f"  Total frames: {total_frames:,}")
    print(f"  Total audio:  {total_frames * FRAME_SIZE / TARGET_SR / 60:.1f} min")
    size = total_frames * FRAME_SIZE * 4 * 2 / 1e9
    print(f"  File size:    {size:.1f} GB")
    print(f"  Saved to:     {DATA_PROC}")
    print("=" * 55)
    print("\n  Next: python src/data/split.py")


if __name__ == "__main__":
    main()