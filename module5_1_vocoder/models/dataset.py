import os
import random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class VocoderPairDataset(Dataset):
    """
    Loads wav_path|mel_path pairs.
    Enforces locked alignment by always deriving audio crop length from mel crop length:
        audio_len_samples = mel_T * hop_length

    Mel can be (T, 80) or (80, T). We normalize to (80, T).
    """
    def __init__(
        self,
        filelist_path: str,
        segment_mel_frames: int = 128,
        sample_rate: int = 16000,
        hop_length: int = 80,
        n_mels: int = 80,
        shuffle: bool = True,
    ):
        self.filelist_path = filelist_path
        self.segment_mel_frames = int(segment_mel_frames)
        self.sr = int(sample_rate)
        self.hop = int(hop_length)
        self.n_mels = int(n_mels)

        self.pairs = []
        with open(filelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '|' not in line:
                    continue
                wav_path, mel_path = line.split('|', 1)
                self.pairs.append((wav_path, mel_path))

        if shuffle:
            random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _normalize_mel(mel: np.ndarray, n_mels: int) -> np.ndarray:
        if mel.ndim != 2:
            raise ValueError(f'mel must be 2D, got shape={mel.shape}')
        # Accept (T,80) or (80,T)
        if mel.shape[0] == n_mels:
            return mel.astype(np.float32)  # (80,T)
        if mel.shape[1] == n_mels:
            return mel.T.astype(np.float32)  # (80,T)
        raise ValueError(f'mel does not contain n_mels={n_mels}: shape={mel.shape}')

    def __getitem__(self, idx: int):
        wav_path, mel_path = self.pairs[idx]

        # Load mel
        mel = np.load(mel_path)
        mel = self._normalize_mel(mel, self.n_mels)  # (80, T)
        T = mel.shape[1]

        # Load audio
        audio, sr = sf.read(wav_path, dtype='float32', always_2d=False)
        if sr != self.sr:
            raise ValueError(f'SR mismatch for {wav_path}: {sr} != {self.sr}')
        if audio.ndim > 1:
            # if stereo, take mean to mono
            audio = audio.mean(axis=1).astype(np.float32)

        # If utterance shorter than segment, pad mel+audio consistently
        seg_T = self.segment_mel_frames
        if T < seg_T:
            pad_T = seg_T - T
            mel = np.pad(mel, ((0,0),(0,pad_T)), mode='edge')
            T = mel.shape[1]

        # Pick mel crop start
        start_T = 0 if T == seg_T else random.randint(0, T - seg_T)
        mel_seg = mel[:, start_T:start_T + seg_T]  # (80, seg_T)

        # LOCKED: audio crop length derived from mel frames
        start_samp = start_T * self.hop
        seg_samp = seg_T * self.hop

        # Some files have a few ms padding mismatch (validated in Step-1). We clamp safely.
        end_samp = start_samp + seg_samp
        if end_samp > len(audio):
            # Shift left if possible; else pad
            shift = end_samp - len(audio)
            start_samp = max(0, start_samp - shift)
            end_samp = start_samp + seg_samp

        audio_seg = audio[start_samp:end_samp]
        if len(audio_seg) < seg_samp:
            audio_seg = np.pad(audio_seg, (0, seg_samp - len(audio_seg)), mode='constant')

        # Return tensors
        # mel: (80, T) float32; audio: (T_samples,) float32
        mel_t = torch.from_numpy(mel_seg)            # (80, seg_T)
        audio_t = torch.from_numpy(audio_seg)        # (seg_samp,)
        return mel_t, audio_t

