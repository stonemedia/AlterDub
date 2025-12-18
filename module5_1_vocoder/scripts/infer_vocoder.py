import os
import argparse
import numpy as np
import torch
import soundfile as sf

from module5_1_vocoder.models.hifigan_generator import Generator


def load_mel_npy(path: str, n_mels: int = 80):
    mel = np.load(path)
    if mel.ndim != 2:
        raise ValueError(f'mel must be 2D, got {mel.shape}')
    if mel.shape[0] == n_mels:
        mel = mel.astype(np.float32)          # (80,T)
    elif mel.shape[1] == n_mels:
        mel = mel.T.astype(np.float32)        # (80,T)
    else:
        raise ValueError(f'mel does not have {n_mels} bins: {mel.shape}')
    return mel


def pick_random_from_filelist(filelist_path: str):
    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip() and '|' in ln]
    wav_path, mel_path = lines[np.random.randint(0, len(lines))].split('|', 1)
    return wav_path, mel_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--out_wav', type=str, required=True)
    ap.add_argument('--mel_npy', type=str, default='')
    ap.add_argument('--filelist', type=str, default='/workspace/AlterDub/module3_dataset/metadata/vocoder_filelist.txt')
    ap.add_argument('--sr', type=int, default=16000)
    ap.add_argument('--mel_frames', type=int, default=1024, help='Number of mel frames to synthesize (128=0.64s, 512=2.56s, 1024=5.12s)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[INFO] device:', device)

    G = Generator(n_mels=80).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    if 'G' in ckpt:
        G.load_state_dict(ckpt['G'])
    else:
        G.load_state_dict(ckpt)
    G.eval()

    if args.mel_npy:
        mel = load_mel_npy(args.mel_npy, n_mels=80)
        src = args.mel_npy
    else:
        wav_path, mel_path = pick_random_from_filelist(args.filelist)
        mel = load_mel_npy(mel_path, n_mels=80)
        src = mel_path

    T = mel.shape[1]
    need = int(args.mel_frames)
    if T < need:
        # pad by repeating edge to reach desired frames
        pad = need - T
        mel = np.pad(mel, ((0,0),(0,pad)), mode='edge')
        T = mel.shape[1]

    start = 0 if T == need else np.random.randint(0, T - need)
    mel_seg = mel[:, start:start+need]

    print('[INFO] mel source:', src)
    print('[INFO] mel segment:', mel_seg.shape, 'frames=', need, 'seconds≈', need*0.005)

    mel_t = torch.from_numpy(mel_seg).unsqueeze(0).to(device)  # (1,80,T)
    with torch.no_grad():
        y = G(mel_t).squeeze(0).squeeze(0).cpu().numpy()

    os.makedirs(os.path.dirname(args.out_wav), exist_ok=True)
    sf.write(args.out_wav, y, args.sr)
    print('[OK] wrote:', args.out_wav, 'len_samples:', len(y), 'seconds≈', len(y)/args.sr)


if __name__ == '__main__':
    main()

