import os
import librosa
import soundfile as sf

# 🔹 CHANGE THIS IF YOUR RAW DATASET FOLDER IS DIFFERENT
RAW_BASE = r"D:\AlterDubDatasets"  # contains female_1, female_2, male_1, male_2

# 🔹 Cleaned output lives inside the AlterDub repo
CLEAN_ROOT = os.path.join("..", "data", "cleaned")

# 🔹 We train & extract features at 16 kHz (matches Module 2)
TARGET_SR = 16000

MIN_DURATION = 1.0    # seconds
MAX_DURATION = 8.0    # seconds

TOP_DB_TRIM = 25      # for trimming leading/trailing silence
TOP_DB_SPLIT = 35     # for splitting internal silence


SPEAKER_MAP = {
    "female_1": "spk_f1",
    "female_2": "spk_f2",
    "male_1":   "spk_m1",
    "male_2":   "spk_m2",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def peak_normalize(audio, peak_db=-1.0):
    """
    Simple peak normalization to a given dBFS.
    peak_db = -1 dBFS means max amplitude ~ 0.89
    """
    import numpy as np

    if audio.size == 0:
        return audio

    peak = float(max(abs(audio.max()), abs(audio.min())))
    if peak == 0.0:
        return audio

    target_linear = 10.0 ** (peak_db / 20.0)
    gain = target_linear / peak
    return audio * gain


def process_file(raw_speaker_name, speaker_id, in_path, out_dir, start_index):
    """
    Load one file, trim, split, filter by duration, normalize, save segments.
    Returns next available index after processing.
    """
    print(f"[{raw_speaker_name} -> {speaker_id}] Processing {in_path}")

    # Load and resample to TARGET_SR, convert to mono
    audio, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)

    # Trim leading/trailing silence
    trimmed, _ = librosa.effects.trim(audio, top_db=TOP_DB_TRIM)

    if trimmed.size == 0:
        print("  -> Skipped (silent after trim)")
        return start_index

    # Split on internal silence
    intervals = librosa.effects.split(trimmed, top_db=TOP_DB_SPLIT)

    if len(intervals) == 0:
        intervals = [(0, trimmed.shape[0])]  # whole signal as one interval

    for (start, end) in intervals:
        segment = trimmed[start:end]
        duration = len(segment) / float(TARGET_SR)

        if duration < MIN_DURATION or duration > MAX_DURATION:
            # Skip segments that are too short or too long
            continue

        # Peak normalize
        segment = peak_normalize(segment, peak_db=-1.0)

        # New standardized filename
        fname = f"{speaker_id}_{start_index:06d}.wav"
        out_path = os.path.join(out_dir, fname)

        sf.write(out_path, segment, TARGET_SR)
        print(f"  -> Saved {out_path} ({duration:.2f} s)")
        start_index += 1

    return start_index


def process_speaker(raw_speaker_name, speaker_id):
    raw_dir = os.path.join(RAW_BASE, raw_speaker_name)
    out_dir = os.path.join(CLEAN_ROOT, speaker_id)

    if not os.path.isdir(raw_dir):
        print(f"[WARN] Raw dir not found for {raw_speaker_name}: {raw_dir}")
        return

    ensure_dir(out_dir)

    index = 1

    # We'll process all wav files in this speaker's folder
    for root, _, files in os.walk(raw_dir):
        for fname in sorted(files, key=lambda x: (len(x), x)):
            if not fname.lower().endswith(".wav"):
                continue

            in_path = os.path.join(root, fname)
            index = process_file(raw_speaker_name, speaker_id, in_path, out_dir, index)

    print(f"[{raw_speaker_name} -> {speaker_id}] Done, total segments: {index - 1}")


def main():
    print("=== AlterDub Module 3 – Cleaning & Renaming Dataset ===")
    print(f"RAW_BASE   : {RAW_BASE}")
    print(f"CLEAN_ROOT : {os.path.abspath(CLEAN_ROOT)}")
    print()

    for raw_name, spk_id in SPEAKER_MAP.items():
        print(f"\n=== Processing speaker: {raw_name} -> {spk_id} ===")
        process_speaker(raw_name, spk_id)

    print("\nAll speakers processed.")


if __name__ == "__main__":
    main()
