import os, random
import numpy as np
import soundfile as sf

FILELIST = "/workspace/AlterDub/module3_dataset/metadata/vocoder_filelist.txt"

# Locked settings
SR = 16000
HOP = 80          # 5 ms @ 16k
N_MELS = 80

def read_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            wav_path, mel_path = line.split("|", 1)
            pairs.append((wav_path, mel_path))
    return pairs

def main():
    pairs = read_pairs(FILELIST)
    print(f"[INFO] Total pairs in filelist: {len(pairs)}")

    sample = random.sample(pairs, k=min(25, len(pairs)))

    for wav_path, mel_path in sample:
        if not os.path.exists(wav_path):
            print(f"[BAD] Missing WAV: {wav_path}")
            continue
        if not os.path.exists(mel_path):
            print(f"[BAD] Missing MEL: {mel_path}")
            continue

        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if sr != SR:
            print(f"[BAD] SR mismatch: {wav_path} sr={sr} expected={SR}")
            continue

        mel = np.load(mel_path)
        if mel.ndim != 2:
            print(f"[BAD] Mel ndim != 2: {mel_path} shape={mel.shape}")
            continue

        if mel.shape[1] == N_MELS:
            T = mel.shape[0]
        elif mel.shape[0] == N_MELS:
            T = mel.shape[1]
        else:
            print(f"[BAD] Mel does not have 80 bins: {mel_path} shape={mel.shape}")
            continue

        expected_len = T * HOP
        actual_len = len(audio)

        diff = abs(actual_len - expected_len)
        diff_ms = diff / SR * 1000.0

        print(f"[OK] {os.path.basename(wav_path)} | audio={actual_len} | mel_T={T} | exp={expected_len} | diff={diff} ({diff_ms:.2f} ms)")

    print("[DONE] Validation finished.")

if __name__ == "__main__":
    main()
