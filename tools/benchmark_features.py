import time
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from module2_feature_extractor.feature_extractor import FeatureConfig, FeatureExtractor

def main():
    cfg = FeatureConfig(sample_rate=16000)
    cfg.finalize()
    extractor = FeatureExtractor(cfg)

    wav_path = REPO_ROOT / "module3_dataset" / "data" / "cleaned" / "spk_m1" / "spk_m1_000012.wav"


    # Load once here so we don't time disk I/O
    y, sr = sf.read(wav_path.as_posix())
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    print(f"Loaded {wav_path}, length: {len(y)/sr:.3f} seconds")

    # Time just wav_to_logmel_and_pitch on in-memory waveform
    t0 = time.time()
    logmel, pitch = extractor.mel_extractor.waveform_to_logmel(y), None
    t1 = time.time()
    print(f"Time for log-mel only: {t1 - t0:.4f} sec, shape={logmel.shape}")

    # If you want to include pitch:
    t2 = time.time()
    logmel2, pitch2 = extractor.wav_to_logmel_and_pitch(wav_path)
    t3 = time.time()
    print(f"Time for log-mel + pitch (incl. load): {t3 - t2:.4f} sec")
    print(f"logmel2 shape={logmel2.shape}, pitch2 shape={pitch2.shape}")

if __name__ == "__main__":
    main()
