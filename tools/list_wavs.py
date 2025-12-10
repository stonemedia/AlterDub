import os
import csv
import wave
import contextlib

# 👉 CHANGE THIS to the parent folder that contains your 4 dataset folders
ROOT_DIR = r"D:\AlterDubDatasets"

# 👉 Name of the CSV file that will be created next to this script
OUTPUT_CSV = "dataset_index.csv"

def get_wav_info(filepath):
    try:
        with contextlib.closing(wave.open(filepath, 'rb')) as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth_bytes = wf.getsampwidth()
            bit_depth = sampwidth_bytes * 8
            duration_sec = n_frames / float(sample_rate) if sample_rate > 0 else 0.0
            return sample_rate, duration_sec, n_channels, bit_depth
    except wave.Error as e:
        print(f"⚠️ Could not read WAV info for: {filepath} ({e})")
        return None, None, None, None

def main():
    rows = []

    for root, dirs, files in os.walk(ROOT_DIR):
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue

            full_path = os.path.join(root, fname)
            rel_folder = os.path.relpath(root, ROOT_DIR)

            sr, dur, ch, bit = get_wav_info(full_path)

            rows.append({
                "folder": rel_folder,
                "file_name": fname,
                "full_path": full_path,
                "sample_rate": sr,
                "duration_sec": dur,
                "channels": ch,
                "bit_depth": bit,
            })

    # Sort nicely: by folder, then file_name
    rows.sort(key=lambda r: (r["folder"], r["file_name"]))

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder",
                "file_name",
                "full_path",
                "sample_rate",
                "duration_sec",
                "channels",
                "bit_depth",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Done. Indexed {len(rows)} files.")
    print(f"📄 CSV saved as: {os.path.abspath(OUTPUT_CSV)}")

if __name__ == "__main__":
    main()
