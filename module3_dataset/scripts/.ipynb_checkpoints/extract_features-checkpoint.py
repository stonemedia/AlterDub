from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# 1. Make repo root importable so we can use module2_feature_extractor
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
MODULE3_ROOT = THIS_FILE.parents[1]       # .../AlterDub/module3_dataset
REPO_ROOT = THIS_FILE.parents[2]          # .../AlterDub

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from module2_feature_extractor.feature_extractor import (  # type: ignore
    FeatureExtractor,
    FeatureConfig,
)

# ------------------------------------------------------------------
# 2. Paths for cleaned WAVs and feature output
# ------------------------------------------------------------------
CLEAN_ROOT = MODULE3_ROOT / "data" / "cleaned"
FEATURE_ROOT = MODULE3_ROOT / "data" / "features"

# You already have these speakers from cleaning step
SPEAKERS = ["spk_f1", "spk_f2", "spk_m1", "spk_m2"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# 3. Core processing
# ------------------------------------------------------------------
def process_file(
    extractor: FeatureExtractor,
    spk_id: str,
    wav_path: Path,
    mel_dir: Path,
    pitch_dir: Path,
) -> None:
    """
    Use Module 2's FeatureExtractor to get log-mel + pitch,
    then save as .npy.
    """
    base = wav_path.stem  # e.g. "spk_f1_000001"
    print(f"[{spk_id}] {base} ...")

    # --- THIS is the important part: reuse Module 2 logic ---
    logmel, pitch_hz = extractor.wav_to_logmel_and_pitch(wav_path)

    # Shapes: logmel = (n_mels, T), pitch_hz = (T,)
    if logmel.shape[1] != pitch_hz.shape[0]:
        raise RuntimeError(
            f"Length mismatch for {wav_path}: "
            f"mel_frames={logmel.shape[1]}, pitch_frames={pitch_hz.shape[0]}"
        )

    # Save .npy files
    mel_out = mel_dir / f"{base}.npy"
    pitch_out = pitch_dir / f"{base}.npy"

    np.save(mel_out, logmel.astype(np.float32))
    np.save(pitch_out, pitch_hz.astype(np.float32))

    print(f"    -> mel:   {mel_out} {logmel.shape}")
    print(f"    -> pitch: {pitch_out} {pitch_hz.shape}")


def process_speaker(extractor: FeatureExtractor, spk_id: str) -> None:
    in_dir = CLEAN_ROOT / spk_id
    if not in_dir.is_dir():
        print(f"[WARN] Clean dir not found for {spk_id}: {in_dir}")
        return

    mel_dir = FEATURE_ROOT / spk_id / "mel"
    pitch_dir = FEATURE_ROOT / spk_id / "pitch"
    ensure_dir(mel_dir)
    ensure_dir(pitch_dir)

    print(f"\n=== Processing speaker: {spk_id} ===")
    print(f"  Input : {in_dir}")
    print(f"  Mel   : {mel_dir}")
    print(f"  Pitch : {pitch_dir}")

    wav_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() == ".wav"])
    print(f"  Found {len(wav_files)} cleaned wav files")

    for wav_path in wav_files:
        process_file(extractor, spk_id, wav_path, mel_dir, pitch_dir)


def main() -> None:
    print("=== AlterDub Module 3 – Feature Extraction via Module 2 ===")
    print(f"Repo root    : {REPO_ROOT}")
    print(f"Cleaned root : {CLEAN_ROOT}")
    print(f"Feature root : {FEATURE_ROOT}")
    print(f"Speakers     : {SPEAKERS}")
    print()

    # --- Use Module 2 config & extractor ---
    cfg = FeatureConfig()
    cfg.finalize()  # ensures hop_length, win_length, fmax, etc. are set
    extractor = FeatureExtractor(cfg)

    for spk in SPEAKERS:
        process_speaker(extractor, spk)

    print("\nAll speakers processed.")


if __name__ == "__main__":
    main()
