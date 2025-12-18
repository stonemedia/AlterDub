import math, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from module5_1_vocoder.models.dataset import VocoderPairDataset

FILELIST = "/workspace/AlterDub/module3_dataset/metadata/vocoder_filelist.txt"
SEG = 128
BS = 16
NW = 4

def is_finite_np(x: np.ndarray) -> bool:
    return np.isfinite(x).all()

def main():
    ds = VocoderPairDataset(FILELIST, segment_mel_frames=SEG)

    # IMPORTANT: no shuffle, deterministic order to find corrupt items
    dl = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=NW, drop_last=False)

    # We'll scan the first N batches; expand if needed
    max_batches = 2000
    for bi, batch in enumerate(dl):
        if bi >= max_batches:
            print(f"[OK] scanned {max_batches} batches, no non-finite values found.")
            return

        mel, aud = batch  # mel: (B,80,SEG), aud: (B, SEG*hop)
        mel_np = mel.numpy()
        aud_np = aud.numpy()

        mel_ok = np.isfinite(mel_np).all()
        aud_ok = np.isfinite(aud_np).all()
        if (not mel_ok) or (not aud_ok):
            print(f"[BAD] batch={bi} mel_finite={mel_ok} aud_finite={aud_ok}")
            # locate exact item(s)
            for j in range(mel_np.shape[0]):
                m_ok = np.isfinite(mel_np[j]).all()
                a_ok = np.isfinite(aud_np[j]).all()
                if (not m_ok) or (not a_ok):
                    idx = bi*BS + j
                    print(f"  -> bad item idx={idx} mel_finite={m_ok} aud_finite={a_ok}")
            return

        # extra: detect extreme audio (all zeros) or huge values
        peak = float(np.max(np.abs(aud_np)))
        if peak > 5.0:  # audio should be ~[-1,1]
            print(f"[SUSPECT] batch={bi} audio peak too high: {peak}")
            for j in range(aud_np.shape[0]):
                p = float(np.max(np.abs(aud_np[j])))
                if p > 5.0:
                    idx = bi*BS + j
                    print(f"  -> suspect item idx={idx} peak={p}")
            return
        if bi % 200 == 0:
            print(f"[SCAN] batch {bi} OK")

if __name__ == "__main__":
    main()
