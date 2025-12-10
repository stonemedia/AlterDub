from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Dict, Any

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
MODULE3_ROOT = THIS_FILE.parents[1]      # .../AlterDub/module3_dataset
META_ROOT = MODULE3_ROOT / "metadata"

MANIFEST_CSV = META_ROOT / "global_manifest.csv"
TRAIN_LIST = META_ROOT / "train_list.txt"
VAL_LIST = META_ROOT / "val_list.txt"

# Train/val ratio
TRAIN_RATIO = 0.9

# Fixed seed for reproducibility
RNG_SEED = 42


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            rows.append(row)
    return rows


def group_by_speaker(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        spk = r["speaker_id"]
        grouped.setdefault(spk, []).append(r)
    return grouped


def write_list(path: Path, entries: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in entries:
            f.write(line + "\n")


def main() -> None:
    print("=== AlterDub Module 3 – Build train/val split ===")
    print(f"Manifest CSV : {MANIFEST_CSV}")
    print(f"Train list   : {TRAIN_LIST}")
    print(f"Val list     : {VAL_LIST}")
    print()

    if not MANIFEST_CSV.is_file():
        print(f"[ERROR] Manifest not found: {MANIFEST_CSV}")
        return

    rows = load_manifest(MANIFEST_CSV)
    print(f"Loaded {len(rows)} rows from manifest")

    grouped = group_by_speaker(rows)
    print(f"Found {len(grouped)} speakers in manifest: {list(grouped.keys())}")

    random.seed(RNG_SEED)

    train_entries: List[str] = []
    val_entries: List[str] = []

    for spk, spk_rows in grouped.items():
        # Shuffle rows for this speaker
        random.shuffle(spk_rows)

        n_total = len(spk_rows)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = n_total - n_train

        spk_train = spk_rows[:n_train]
        spk_val = spk_rows[n_train:]

        print(f"[{spk}] total={n_total}, train={n_train}, val={n_val}")

        for r in spk_train:
            mel_path = r["mel_path"]
            pitch_path = r["pitch_path"]
            spk_idx = r["speaker_index"]
            n_frames = r["n_frames"]

            line = f"{mel_path}|{pitch_path}|{spk_idx}|{n_frames}"
            train_entries.append(line)

        for r in spk_val:
            mel_path = r["mel_path"]
            pitch_path = r["pitch_path"]
            spk_idx = r["speaker_index"]
            n_frames = r["n_frames"]

            line = f"{mel_path}|{pitch_path}|{spk_idx}|{n_frames}"
            val_entries.append(line)

    # Shuffle global order (optional but nice)
    random.shuffle(train_entries)
    random.shuffle(val_entries)

    write_list(TRAIN_LIST, train_entries)
    write_list(VAL_LIST, val_entries)

    print()
    print(f"Wrote {len(train_entries)} lines to {TRAIN_LIST}")
    print(f"Wrote {len(val_entries)} lines to {VAL_LIST}")


if __name__ == "__main__":
    main()
