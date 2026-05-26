def collect_speech_files(fraction: float) -> list:
    # Kaggle mounted path
    kaggle_path = Path("/kaggle/input/datasets/yesha1910/librispeech/LibriSpeech/train-clean-100")
    # Local LG Gram path
    local_path  = DATA_RAW / "librispeech" / "LibriSpeech" / "train-clean-100"

    if kaggle_path.exists():
        speech_root = kaggle_path
        print(f"  Using Kaggle LibriSpeech")
    elif local_path.exists():
        speech_root = local_path
        print(f"  Using local LibriSpeech")
    else:
        raise FileNotFoundError("LibriSpeech not found in Kaggle or local path")

    files = list(speech_root.rglob("*.flac"))
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
    # Local synthetic noise
    local_synthetic = DATA_RAW / "musan_synthetic"
    # Local downloaded MUSAN
    local_musan = DATA_RAW / "musan"

    if kaggle_musan.exists():
        noise_files.extend(list(kaggle_musan.rglob("*.wav")))
        print(f"  Using Kaggle MUSAN noise")
    elif local_musan.exists():
        noise_files.extend(list((local_musan / "noise").rglob("*.wav")))
        print(f"  Using local MUSAN noise")
    elif local_synthetic.exists():
        noise_files.extend(list(local_synthetic.rglob("*.wav")))
        print(f"  Using synthetic noise")
    else:
        raise FileNotFoundError("No noise files found")

    random.seed(SEED + 1)
    random.shuffle(noise_files)
    print(f"  Noise files: {len(noise_files)}")
    return noise_files