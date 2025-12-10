from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# 1. Resolve paths
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
MODULE3_ROOT = THIS_FILE.parents[1]      # .../AlterDub/module3_dataset
REPO_ROOT = THIS_FILE.parents[2]         # .../AlterDub

CLEAN_ROOT = MODULE3_ROOT / "data" / "cleaned"
FEATURE_ROOT = MODULE3_ROOT / "data" / "features"
META_ROOT = MODULE3_ROOT / "metadata"

SPEAKERS = ["spk_f1", "spk_f2", "spk_m1", "spk_m2"]
SPEAKER_TO_INDEX = {spk: i for i, spk in enumerate(SPEAKERS)}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# 2. Build manifest rows
# ------------------------------------------------------------------
def build_rows() -> list[dict]:
    rows: list[dict] = []

    for spk in SPEAKERS:
        spk_idx = SPEAKER_TO_INDEX[spk]

        clean_dir = CLEAN_ROOT / spk
        mel_dir = FEATURE_ROOT / spk / "mel"
        pitch_dir = FEATURE_ROOT / spk / "pitch"

        if not clean_dir.is_dir():
            print(f"[WARN] Missing cleaned dir for {spk}: {clean_dir}")
            continue
        if not mel_dir.is_dir():
            print(f"[WARN] Missing mel dir for {spk}: {mel_dir}")
            continue
        if not pitch_dir.is_dir():
            print(f"[WARN] Missing pitch dir for {spk}: {pitch_dir}")
            continue

        wav_files = sorted([p for p in clean_dir.iterdir() if p.suffix.lower() == ".wav"])
        print(f"[{spk}] Found {len(wav_files)} cleaned wav files")

        for wav_path in wav_files:
            base = wav_path.stem  # e.g. "spk_m2_002441"

            mel_path = mel_dir / f"{base}.npy"
            pitch_path = pitch_dir / f"{base}.npy"

            if not mel_path.is_file():
                print(f"  [SKIP] mel not found for {base}: {mel_path}")
                continue
            if not pitch_path.is_file():
                print(f"  [SKIP] pitch not found for {base}: {pitch_path}")
                continue

            # Load mel to get number of frames
            mel = np.load(mel_path)
            if mel.ndim != 2 or mel.shape[0] != 80:
                print(f"  [WARN] Unexpected mel shape for {base}: {mel.shape}")
            n_frames = int(mel.shape[1])

            # Approx duration from frames (5 ms hop)
            duration_sec = n_frames * 0.005  # 5 ms per frame

            # Store paths relative to repo root to make training portable
            wav_rel = wav_path.relative_to(REPO_ROOT)
            mel_rel = mel_path.relative_to(REPO_ROOT)
            pitch_rel = pitch_path.relative_to(REPO_ROOT)

            row = {
                "utt_id": base,
                "speaker_id": spk,
                "speaker_index": spk_idx,
                "wav_path": str(wav_rel).replace("\\", "/"),
                "mel_path": str(mel_rel).replace("\\", "/"),
                "pitch_path": str(pitch_rel).replace("\\", "/"),
                "n_frames": n_frames,
                "duration_sec": f"{duration_sec:.3f}",
            }
            rows.append(row)

    return rows


# ------------------------------------------------------------------
# 3. Write CSV
# ------------------------------------------------------------------
def write_csv(rows: list[dict], out_path: Path) -> None:
    ensure_dir(out_path.parent)

    fieldnames = [
        "utt_id",
        "speaker_id",
        "speaker_index",
        "wav_path",
        "mel_path",
        "pitch_path",
        "n_frames",
        "duration_sec",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nWrote {len(rows)} rows to {out_path}")


def main() -> None:
    print("=== AlterDub Module 3 – Build Global Manifest ===")
    print(f"Repo root   : {REPO_ROOT}")
    print(f"Cleaned dir : {CLEAN_ROOT}")
    print(f"Feature dir : {FEATURE_ROOT}")
    print(f"Meta dir    : {META_ROOT}")
    print(f"Speakers    : {SPEAKERS}")
    print()

    rows = build_rows()
    if not rows:
        print("No rows generated, check directories.")
        return

    out_csv = META_ROOT / "global_manifest.csv"
    write_csv(rows, out_csv)


if __name__ == "__main__":
    main()
