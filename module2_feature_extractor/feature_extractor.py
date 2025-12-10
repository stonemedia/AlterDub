# module2_feature_extractor/feature_extractor.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import librosa
import numpy as np
import json
import soundfile as sf


@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    n_mels: int = 80
    fmin: float = 0.0
    fmax: Optional[float] = None
    hop_length: Optional[int] = None
    win_length: Optional[int] = None

    pitch_fmin: float = 100.0
    pitch_fmax: float = 400.0

    def finalize(self) -> None:
        if self.win_length is None:
            self.win_length = int(0.010 * self.sample_rate)
        if self.hop_length is None:
            self.hop_length = int(0.005 * self.sample_rate)
        if self.fmax is None:
            self.fmax = self.sample_rate / 2


class MelExtractor:
    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self.n_fft = cfg.n_fft
        self.hop = cfg.hop_length
        self.win = cfg.win_length
        self.n_mels = cfg.n_mels

        self.window = np.hanning(self.win).astype(np.float32)

        self.mel_basis = librosa.filters.mel(
            sr=cfg.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
        ).astype(np.float32)

    @staticmethod
    def _power_to_db(mel_power: np.ndarray, top_db: float = None) -> np.ndarray:
        amin = 1e-10
        ref = 1.0

        log_spec = 10.0 * np.log10(np.maximum(mel_power, amin) / ref)

        if top_db is not None:
            max_val = np.max(log_spec)
            log_spec = np.maximum(log_spec, max_val - top_db)

        return log_spec.astype(np.float32)

    def waveform_to_logmel(self, y: np.ndarray) -> np.ndarray:
        if y.ndim != 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)

        n_samples = y.shape[0]
        if n_samples < self.win:
            return np.zeros((self.n_mels, 0), dtype=np.float32)

        n_frames = 1 + (n_samples - self.win) // self.hop
        spec = np.empty((self.n_fft // 2 + 1, n_frames), dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop
            end = start + self.win
            frame = y[start:end] * self.window

            if self.n_fft > self.win:
                frame_padded = np.pad(frame, (0, self.n_fft - self.win))
            else:
                frame_padded = frame

            fft = np.fft.rfft(frame_padded, n=self.n_fft)
            power = np.abs(fft) ** 2
            spec[:, i] = power.astype(np.float32)

        mel_power = np.dot(self.mel_basis, spec)
        logmel = self._power_to_db(mel_power, top_db=None)
        return logmel


class FeatureExtractor:
    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        if config is None:
            config = FeatureConfig()
        config.finalize()

        self.cfg = config
        self.mel_extractor = MelExtractor(config)

    def _load_wav_aligned(self, wav_path: Path) -> np.ndarray:
        y, sr = sf.read(wav_path.as_posix())

        if y.ndim > 1:
            y = y.mean(axis=1)

        y = y.astype(np.float32)

        if sr != self.cfg.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.cfg.sample_rate)

        return y

    def wav_to_logmel(self, wav_path: str | Path) -> np.ndarray:
        y = self._load_wav_aligned(Path(wav_path))
        return self.mel_extractor.waveform_to_logmel(y)

    def wav_to_logmel_and_pitch(self, wav_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        y = self._load_wav_aligned(Path(wav_path))

        logmel = self.mel_extractor.waveform_to_logmel(y)

        pitch_frame_length = max(self.cfg.n_fft, self.cfg.win_length, 1024)

        pitch_hz = librosa.yin(
            y=y,
            fmin=self.cfg.pitch_fmin,
            fmax=self.cfg.pitch_fmax,
            sr=self.cfg.sample_rate,
            frame_length=pitch_frame_length,
            hop_length=self.cfg.hop_length,
        ).astype(np.float32)

        pitch_hz = np.nan_to_num(pitch_hz, nan=0.0)

        T_mel = logmel.shape[1]
        T_pitch = pitch_hz.shape[0]
        if T_pitch > T_mel:
            pitch_hz = pitch_hz[:T_mel]
        else:
            pitch_hz = np.pad(pitch_hz, (0, T_mel - T_pitch), constant_values=0.0)

        return logmel, pitch_hz
