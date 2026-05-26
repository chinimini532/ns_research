"""
src/data/download.py

Downloads LibriSpeech, MUSAN, and DEMAND datasets.

Usage:
    python src/data/download.py --fraction 0.02   # LG Gram - fast test (~300MB)
    python src/data/download.py --fraction 1.0    # Kaggle  - full dataset

Fraction logic:
    fraction < 1.0 → downloads LibriSpeech dev-clean (337MB) for testing
    fraction = 1.0 → downloads LibriSpeech train-clean-100 (6.3GB) for training

    MUSAN and DEMAND always download only the fraction needed.
"""

import os
import sys
import argparse
import tarfile
import requests
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm
import random

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.config import DATA_RAW, FRACTION

# ─── Dataset URLs ─────────────────────────────────────────────────────────────

# Small subset for development (337MB) — used when fraction < 1.0
LIBRISPEECH_DEV_URL   = "https://www.openslr.org/resources/12/dev-clean.tar.gz"

# Full training set (6.3GB) — used when fraction = 1.0 on Kaggle
LIBRISPEECH_TRAIN_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"

DEMAND_URLS = {
    "DKITCHEN": "https://zenodo.org/record/1227121/files/DKITCHEN_16k.zip",
    "OOFFICE":  "https://zenodo.org/record/1227121/files/OOFFICE_16k.zip",
    "STRAFFIC": "https://zenodo.org/record/1227121/files/STRAFFIC_16k.zip",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = "") -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    filepath = dest / filename

    if filepath.exists():
        print(f"  Already exists: {filename} — skipping")
        return filepath

    print(f"  Downloading: {desc or filename}")
    response = requests.get(url, stream=True, timeout=60)
    total = int(response.headers.get("content-length", 0))

    with open(filepath, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        unit_divisor=1024, desc=filename[:40]
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return filepath


def extract_tar(filepath: Path, dest: Path):
    marker = dest / f".extracted_{filepath.stem}"
    if marker.exists():
        print(f"  Already extracted: {filepath.name}")
        return
    print(f"  Extracting: {filepath.name} ...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(dest)
    marker.touch()
    filepath.unlink()
    print(f"  Done.")


def extract_zip(filepath: Path, dest: Path):
    marker = dest / f".extracted_{filepath.stem}"
    if marker.exists():
        print(f"  Already extracted: {filepath.name}")
        return
    print(f"  Extracting: {filepath.name} ...")
    with zipfile.ZipFile(filepath, "r") as z:
        z.extractall(dest)
    marker.touch()
    filepath.unlink()
    print(f"  Done.")


def apply_fraction(directory: Path, extensions: list, fraction: float, seed: int = 42):
    if fraction >= 1.0:
        return
    all_files = []
    for ext in extensions:
        all_files.extend(list(directory.rglob(f"*{ext}")))
    if not all_files:
        return
    random.seed(seed)
    random.shuffle(all_files)
    keep_count = max(10, int(len(all_files) * fraction))
    files_to_delete = all_files[keep_count:]
    print(f"  Keeping {keep_count}/{len(all_files)} files (fraction={fraction})")
    for f in tqdm(files_to_delete, desc="  Pruning"):
        f.unlink()
    for dirpath in sorted(directory.rglob("*"), reverse=True):
        if dirpath.is_dir():
            try:
                dirpath.rmdir()
            except OSError:
                pass


# ─── Dataset Functions ────────────────────────────────────────────────────────

def download_librispeech(fraction: float):
    print("\n[1/3] LibriSpeech")
    print("-" * 40)

    dest = DATA_RAW / "librispeech"

    if fraction < 1.0:
        # Use small dev-clean subset for LG Gram testing
        url      = LIBRISPEECH_DEV_URL
        sub_name = "dev-clean"
        print(f"  Mode: dev-clean (337MB) — development/testing")
    else:
        # Use full training set for Kaggle
        url      = LIBRISPEECH_TRAIN_URL
        sub_name = "train-clean-100"
        print(f"  Mode: train-clean-100 (6.3GB) — full training")

    speech_dir = dest / "LibriSpeech" / sub_name

    if speech_dir.exists() and any(speech_dir.rglob("*.flac")):
        count = len(list(speech_dir.rglob("*.flac")))
        print(f"  Already downloaded: {count} .flac files")
    else:
        filepath = download_file(url, dest, f"LibriSpeech {sub_name}")
        extract_tar(filepath, dest)

    # Apply fraction only for dev-clean (it's already small but trim further)
    if fraction < 1.0:
        apply_fraction(speech_dir, [".flac"], fraction)

    count = len(list(speech_dir.rglob("*.flac")))
    print(f"  Final: {count} speech files")
    return speech_dir


def download_musan(fraction: float):
    print("\n[2/3] MUSAN")
    print("-" * 40)

    dest     = DATA_RAW / "musan"
    musan_dir = dest / "musan"

    if musan_dir.exists() and any(musan_dir.rglob("*.wav")):
        count = len(list(musan_dir.rglob("*.wav")))
        print(f"  Already downloaded: {count} .wav files")
    else:
        filepath = download_file(MUSAN_URL, dest, "MUSAN")
        extract_tar(filepath, dest)

    for subdir in ["noise", "music"]:
        subpath = musan_dir / subdir
        if subpath.exists():
            apply_fraction(subpath, [".wav"], fraction)

    count = len(list(musan_dir.rglob("*.wav")))
    print(f"  Final: {count} noise files")


def download_demand(fraction: float):
    print("\n[3/3] DEMAND")
    print("-" * 40)

    dest = DATA_RAW / "demand"
    dest.mkdir(parents=True, exist_ok=True)

    for env_name, url in DEMAND_URLS.items():
        env_dir = dest / env_name
        if env_dir.exists() and any(env_dir.rglob("*.wav")):
            print(f"  Already downloaded: {env_name}")
            continue
        try:
            filepath = download_file(url, dest, f"DEMAND {env_name}")
            extract_zip(filepath, dest)
        except Exception as e:
            print(f"  Warning: {env_name} failed: {e}")

    apply_fraction(dest, [".wav"], fraction)
    count = len(list(dest.rglob("*.wav")))
    print(f"  Final: {count} room noise files")


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(fraction: float):
    print("\n" + "=" * 55)
    print("  Download Summary")
    print("=" * 55)

    sub = "dev-clean" if fraction < 1.0 else "train-clean-100"
    speech_dir = DATA_RAW / "librispeech" / "LibriSpeech" / sub
    musan_dir  = DATA_RAW / "musan" / "musan"
    demand_dir = DATA_RAW / "demand"

    s = len(list(speech_dir.rglob("*.flac"))) if speech_dir.exists() else 0
    m = len(list(musan_dir.rglob("*.wav")))   if musan_dir.exists() else 0
    d = len(list(demand_dir.rglob("*.wav")))  if demand_dir.exists() else 0

    print(f"  LibriSpeech speech:  {s:,} files")
    print(f"  MUSAN noise:         {m:,} files")
    print(f"  DEMAND room noise:   {d:,} files")
    print(f"  Total noise:         {m+d:,} files")
    print("=" * 55)

    if s > 0 and (m + d) > 0:
        print("\n  All datasets ready.")
        print("  Next: python src/data/preprocess.py")
    else:
        print("\n  WARNING: Some datasets missing.")

    return s, m, d


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fraction", type=float, default=FRACTION,
        help="Fraction to keep. <1.0 uses dev-clean, =1.0 uses train-clean-100"
    )
    parser.add_argument(
        "--skip-demand", action="store_true",
        help="Skip DEMAND download"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  NS Research — Dataset Download")
    print("=" * 55)
    print(f"  Fraction: {args.fraction}")
    print(f"  Mode:     {'Development (LG Gram)' if args.fraction < 1.0 else 'Full training (Kaggle)'}")
    print(f"  Save to:  {DATA_RAW}")
    print("=" * 55)

    download_librispeech(args.fraction)
    download_musan(args.fraction)

    if not args.skip_demand:
        download_demand(args.fraction)
    else:
        print("\n[3/3] DEMAND — skipped")

    print_summary(args.fraction)


if __name__ == "__main__":
    main()