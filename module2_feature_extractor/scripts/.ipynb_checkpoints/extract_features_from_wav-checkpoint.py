# module2_feature_extractor/scripts/extract_features_from_wav.py

import argparse
import time
import json
import numpy as np
from pathlib import Path

from module2_feature_extractor.feature_extractor import FeatureConfig, FeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract log-mel, pitch, or full feature bundle (Module 2 v0.3)"
    )

    parser.add_argument(
        "wav_path",
        type=str,
        help="Path to input WAV file"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="module2_feature_extractor/data/features",
        help="Root directory where features will be saved"
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default=None,
        help="Optional speaker/voice id (e.g., spk001)"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate to resample audio to (default: 16000)"
    )

    # v0.2
    parser.add_argument(
        "--with_pitch",
        action="store_true",
        help="If set, also extract and save pitch features",
    )

    # v0.3
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="If set, save full feature bundle (.npz) instead of separate .npy files",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = FeatureConfig(sample_rate=args.sample_rate)
    extractor = FeatureExtractor(config=cfg)

    wav_path = Path(args.wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    t0 = time.time()

    # ----------------------------------------------------------------------
    # v0.3 FEATURE BUNDLE  (logmel + logmel_norm + pitch + meta)
    # ----------------------------------------------------------------------
    if args.bundle:
        out_path = extractor.save_feature_bundle(
            wav_path=wav_path,
            out_root=args.out_root,
            speaker_id=args.speaker_id,
        )
        t1 = time.time()

        data = np.load(out_path, allow_pickle=True)
        logmel = data["logmel"]
        logmel_norm = data["logmel_norm"]
        pitch = data["pitch_hz"]
        meta = json.loads(str(data["meta"]))

        print(f"Saved feature bundle to: {out_path}")
        print(f"log-mel shape     : {logmel.shape} (n_mels, T)")
        print(f"log-mel norm shape: {logmel_norm.shape} (n_mels, T)")
        print(f"pitch shape       : {pitch.shape} (T,)")
        print(f"meta              : {meta}")
        print(f"Extraction time (bundle): {t1 - t0:.4f} seconds")
        return

    # ----------------------------------------------------------------------
    # v0.2 LOG-MEL + PITCH (separate .npy files)
    # ----------------------------------------------------------------------
    if args.with_pitch:
        logmel_path, pitch_path = extractor.save_logmel_and_pitch(
            wav_path=wav_path,
            out_root=args.out_root,
            speaker_id=args.speaker_id,
        )
        t1 = time.time()

        logmel = np.load(logmel_path)
        pitch = np.load(pitch_path)

        print(f"Saved log-mel to : {logmel_path}")
        print(f"Saved pitch to   : {pitch_path}")
        print(f"log-mel shape: {logmel.shape} (n_mels, T)")
        print(f"pitch shape  : {pitch.shape} (T,)")
        print(f"Extraction time (logmel + pitch): {t1 - t0:.4f} seconds")
        return

    # ----------------------------------------------------------------------
    # v0.1 LOG-MEL ONLY
    # ----------------------------------------------------------------------
    out_path = extractor.save_logmel(
        wav_path=wav_path,
        out_root=args.out_root,
        speaker_id=args.speaker_id,
    )
    t1 = time.time()

    feat = np.load(out_path)

    print(f"Saved features to: {out_path}")
    print(f"Feature shape: {feat.shape} (n_mels, T)")
    print(f"dtype: {feat.dtype}")
    print(f"Extraction time (logmel only): {t1 - t0:.4f} seconds")


if __name__ == "__main__":
    main()
