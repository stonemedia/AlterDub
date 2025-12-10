# module2_feature_extractor/streaming_feature_extractor.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from module2_feature_extractor.feature_extractor import FeatureConfig, MelExtractor


@dataclass
class StreamingState:
    next_frame_index: int = 0
    pcm_buffer: Optional[np.ndarray] = None


class StreamingFeatureExtractor:
    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        if config is None:
            config = FeatureConfig()
        config.finalize()

        self.cfg = config
        self.mel_extractor = MelExtractor(config)
        self.state = StreamingState()

    def _append_pcm(self, pcm: np.ndarray) -> None:
        if pcm.ndim != 1:
            raise ValueError("PCM must be 1D mono")

        if np.issubdtype(pcm.dtype, np.integer):
            pcm = pcm.astype(np.float32) / 32768.0
        else:
            pcm = pcm.astype(np.float32)

        if self.state.pcm_buffer is None:
            self.state.pcm_buffer = pcm
        else:
            self.state.pcm_buffer = np.concatenate([self.state.pcm_buffer, pcm], axis=0)

    def _compute_new_frames(self) -> np.ndarray:
        y = self.state.pcm_buffer
        if y is None:
            return np.zeros((self.cfg.n_mels, 0), dtype=np.float32)

        n_samples = y.shape[0]
        win = self.cfg.win_length
        hop = self.cfg.hop_length
        n_fft = self.cfg.n_fft

        if n_samples < win:
            return np.zeros((self.cfg.n_mels, 0), dtype=np.float32)

        total_frames = 1 + (n_samples - win) // hop
        start_frame = self.state.next_frame_index

        if total_frames <= start_frame:
            return np.zeros((self.cfg.n_mels, 0), dtype=np.float32)

        n_new = total_frames - start_frame
        spec_new = np.empty((n_fft // 2 + 1, n_new), dtype=np.float32)

        window = self.mel_extractor.window

        for i in range(n_new):
            frame_idx = start_frame + i
            s = frame_idx * hop
            e = s + win
            frame = y[s:e] * window

            if n_fft > win:
                frame_padded = np.pad(frame, (0, n_fft - win))
            else:
                frame_padded = frame

            fft = np.fft.rfft(frame_padded, n=n_fft)
            spec_new[:, i] = (np.abs(fft) ** 2).astype(np.float32)

        mel_power = np.dot(self.mel_extractor.mel_basis, spec_new)
        logmel = self.mel_extractor._power_to_db(mel_power, top_db=None)

        self.state.next_frame_index = total_frames
        return logmel

    def process_chunk(self, pcm_chunk: np.ndarray) -> np.ndarray:
        self._append_pcm(pcm_chunk)
        return self._compute_new_frames()
