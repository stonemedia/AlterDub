import os
import subprocess
import numpy as np

# ---- CONFIG ----
ALTERDUB_ROOT = "/workspace/AlterDub"
INFER_SCRIPT = f"{ALTERDUB_ROOT}/module5_1_vocoder/scripts/infer_vocoder.py"
CKPT_DIR     = f"{ALTERDUB_ROOT}/module5_1_vocoder/checkpoints"
MEL_ROOT     = f"{ALTERDUB_ROOT}/module3_dataset/data/features"
OUT_DIR      = f"{ALTERDUB_ROOT}/module5_1_vocoder/outputs/eval_v2_ckpt_sweep"

# Your selected utterances
UTTS = [
    "spk_f1_000248", "spk_f1_000001", "spk_f1_000264",
    "spk_f2_001153", "spk_f2_000426", "spk_f2_000183",
    "spk_m1_001839", "spk_m1_002142", "spk_m1_001412",
    "spk_m2_000283", "spk_m2_000049", "spk_m2_001937",
]

# Checkpoints you want to evaluate (edit this list as needed)
STEPS = [
    50000, 60000, 70000, 80000, 90000,
    95000, 96000, 97000, 98000, 99000,
    100000
]

# Checkpoint filename template (adjust if your naming differs)
CKPT_TEMPLATE = "vocoder_v2_hifigan_ultrasafe_mel512_b4_step{step}.pt"

# ---- HELPERS ----
def mel_path_for_utt(utt_id: str) -> str:
    spk = utt_id.split("_")[0]  # spk_f1, spk_m2 etc.
    return f"{MEL_ROOT}/{spk}/mel/{utt_id}.npy"

def mel_frames(mel_npy_path: str, n_mels: int = 80) -> int:
    m = np.load(mel_npy_path)
    if m.ndim != 2:
        raise ValueError(f"Mel must be 2D. Got {m.shape} for {mel_npy_path}")
    # accept (80,T) or (T,80)
    if m.shape[0] == n_mels:
        return int(m.shape[1])
    if m.shape[1] == n_mels:
        return int(m.shape[0])
    raise ValueError(f"Mel shape doesn't match n_mels={n_mels}: {m.shape} for {mel_npy_path}")

def run(cmd: list):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

# ---- MAIN ----
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for step in STEPS:
        ckpt = f"{CKPT_DIR}/{CKPT_TEMPLATE.format(step=step)}"
        if not os.path.isfile(ckpt):
            print(f"[SKIP] Missing ckpt: {ckpt}")
            continue

        step_out = os.path.join(OUT_DIR, f"step{step}")
        os.makedirs(step_out, exist_ok=True)

        for utt in UTTS:
            mel = mel_path_for_utt(utt)
            if not os.path.isfile(mel):
                print(f"[SKIP] Missing mel: {mel}")
                continue

            T = mel_frames(mel)
            out_wav = os.path.join(step_out, f"V2_step{step}_{utt}.wav")

            cmd = [
                "python", INFER_SCRIPT,
                "--ckpt", ckpt,
                "--mel_npy", mel,
                "--out_wav", out_wav,
                "--mel_frames", str(T),
            ]

            # Run inference
            try:
                run(cmd)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] step={step} utt={utt}: {e}")

    print(f"\nDone. Outputs in: {OUT_DIR}")

if __name__ == "__main__":
    main()
