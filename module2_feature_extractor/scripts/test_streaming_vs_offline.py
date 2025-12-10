# module2_feature_extractor/scripts/test_streaming_vs_offline.py

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

from module2_feature_extractor.feature_extractor import FeatureConfig, MelExtractor
from module2_feature_extractor.streaming_feature_extractor import StreamingFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare offline vs streaming log-mel extraction (numpy STFT)."
    )
    parser.add_argument(
        "wav_path",
        type=str,
        help="Path to input WAV file",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate to resample audio to (default: 16000)",
    )
    parser.add_argument(
        "--chunk_ms",
        type=float,
        default=10.0,
        help="Chunk size in milliseconds for streaming (default: 10 ms)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = FeatureConfig(sample_rate=args.sample_rate)
    cfg.finalize()

    streaming_extractor = StreamingFeatureExtractor(config=cfg)
    mel_extractor = MelExtractor(cfg)

    wav_path = Path(args.wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    # ---- Load WAV once ----
    y, sr = sf.read(wav_path.as_posix())
    if y.ndim > 1:
        y = y.mean(axis=1)  # mono
    if sr != cfg.sample_rate:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=cfg.sample_rate)
        sr = cfg.sample_rate
    else:
        y = y.astype(np.float32)

    # ---- Offline reference using SAME MelExtractor ----
    logmel_offline = mel_extractor.waveform_to_logmel(y)
    print(f"Offline log-mel shape: {logmel_offline.shape}")

    # ---- Streaming simulation ----
    chunk_size_samples = int(args.chunk_ms * 0.001 * sr)
    all_stream_frames = []

    for start in range(0, len(y), chunk_size_samples):
        end = min(start + chunk_size_samples, len(y))
        chunk = y[start:end]

        new_frames = streaming_extractor.process_chunk(chunk)
        if new_frames.shape[1] > 0:
            all_stream_frames.append(new_frames)

    if len(all_stream_frames) > 0:
        logmel_streaming = np.concatenate(all_stream_frames, axis=1)
    else:
        logmel_streaming = np.zeros((cfg.n_mels, 0), dtype=np.float32)

    print(f"Streaming log-mel shape: {logmel_streaming.shape}")

    # ---- Compare shapes ----
    min_T = min(logmel_offline.shape[1], logmel_streaming.shape[1])
    off = logmel_offline[:, :min_T]
    stream = logmel_streaming[:, :min_T]

    diff = np.abs(off - stream)
    print(f"Mean abs diff (offline vs streaming, first {min_T} frames): {diff.mean():.6f}")
    print(f"Max  abs diff (offline vs streaming, first {min_T} frames): {diff.max():.6f}")


if __name__ == "__main__":
    main()
