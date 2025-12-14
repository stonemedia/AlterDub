import os
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class VCDataset(Dataset):
    """
    Voice Conversion Dataset for Module 4.

    Expects a list file where each line has the format:
        <mel_path>|<pitch_path>|<speaker_id>|<num_frames>

    - mel_path: path to .npy file with shape (n_mels, T)
    - pitch_path: path to .npy file with shape (T,)
    - speaker_id: integer id for the speaker
    - num_frames: integer number of frames (can be used for sanity checks)

    This dataset:
    - Loads mel and f0
    - Converts mel to shape (T, n_mels)
    - Optionally crops to a fixed segment length (segment_frames) for training
    """

    def __init__(
        self,
        list_path: str,
        segment_frames: int,
        is_train: bool = True,
    ) -> None:
        """
        Args:
            list_path: Path to train_list.txt or val_list.txt
            segment_frames: Number of frames to crop per sample
            is_train: If True, enables random cropping; otherwise uses simple cropping (first segment)
        """
        super().__init__()
        self.list_path = list_path
        self.segment_frames = segment_frames
        self.is_train = is_train

        if not os.path.isfile(self.list_path):
            raise FileNotFoundError(f"List file not found: {self.list_path}")

        self.entries = self._load_entries(self.list_path)

    @staticmethod
    def _load_entries(list_path: str) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue

                parts = line.split("|")
                if len(parts) < 4:
                    raise ValueError(f"Invalid line in list file: {line}")

                mel_path, pitch_path, spk_id_str, num_frames_str = parts[:4]

                entry = {
                    "mel_path": mel_path,
                    "pitch_path": pitch_path,
                    "spk_id": int(spk_id_str),
                    "num_frames": int(num_frames_str),
                }
                entries.append(entry)

        if len(entries) == 0:
            raise ValueError(f"No valid entries found in list file: {list_path}")

        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def _load_feature_pair(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        mel_path = entry["mel_path"]
        pitch_path = entry["pitch_path"]
        spk_id = entry["spk_id"]
        num_frames_meta = entry["num_frames"]

        if not os.path.isfile(mel_path):
            raise FileNotFoundError(f"Mel file not found: {mel_path}")
        if not os.path.isfile(pitch_path):
            raise FileNotFoundError(f"Pitch file not found: {pitch_path}")

        mel_np = np.load(mel_path)   # expected shape: (n_mels, T)
        f0_np = np.load(pitch_path)  # expected shape: (T,)

        if mel_np.ndim != 2:
            raise ValueError(f"Mel file {mel_path} must be 2D (n_mels, T), got shape {mel_np.shape}")
        if f0_np.ndim != 1:
            raise ValueError(f"Pitch file {pitch_path} must be 1D (T,), got shape {f0_np.shape}")

        n_mels, T_mel = mel_np.shape
        T_f0 = f0_np.shape[0]

        if T_mel != T_f0:
            # We allow small mismatch but warn; for now, truncate to min length
            T_min = min(T_mel, T_f0)
            mel_np = mel_np[:, :T_min]
            f0_np = f0_np[:T_min]
            T_mel = T_min
            T_f0 = T_min

        # Optional sanity check with num_frames from list file
        if abs(T_mel - num_frames_meta) > 2:
            # Just a warning; not fatal
            # (we don't print here to avoid spam; you can add logging later)
            pass

        # Convert mel to (T, n_mels)
        mel_np = mel_np.T  # (T, n_mels)

        return {
            "mel": mel_np,
            "f0": f0_np,
            "spk_id": spk_id,
            "num_frames": T_mel,
        }

    def _random_or_fixed_crop(
        self,
        mel: np.ndarray,
        f0: np.ndarray,
        segment_frames: int,
        is_train: bool,
    ) -> (np.ndarray, np.ndarray):
        """
        Crop (mel, f0) along time axis to segment_frames.
        - If utterance shorter than segment_frames: return full sequence (no padding here).
        - If longer:
          - training: random crop
          - eval: use first segment (or you could use center crop if you prefer)
        """
        T = mel.shape[0]
        if T <= segment_frames:
            return mel, f0

        if is_train:
            # random crop
            max_start = T - segment_frames
            start = np.random.randint(0, max_start + 1)
        else:
            # deterministic crop (start from 0 for now)
            start = 0

        end = start + segment_frames
        mel_seg = mel[start:end, :]
        f0_seg = f0[start:end]

        return mel_seg, f0_seg

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        feat = self._load_feature_pair(entry)

        mel_np: np.ndarray = feat["mel"]     # (T, n_mels)
        f0_np: np.ndarray = feat["f0"]       # (T,)
        spk_id: int = feat["spk_id"]

        # Apply crop if segment_frames is set (> 0)
        if self.segment_frames is not None and self.segment_frames > 0:
            mel_np, f0_np = self._random_or_fixed_crop(
                mel_np, f0_np, self.segment_frames, self.is_train
            )

        T = mel_np.shape[0]

        # Convert to torch tensors
        mel = torch.from_numpy(mel_np).float()      # (T, n_mels)
        f0 = torch.from_numpy(f0_np).float()        # (T,)
        spk_id_tensor = torch.tensor(spk_id, dtype=torch.long)
        length_tensor = torch.tensor(T, dtype=torch.long)

        return {
            "mel": mel,
            "f0": f0,
            "spk_id": spk_id_tensor,
            "length": length_tensor,
        }


def vc_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for VCDataset.

    Pads sequences in the batch to the max length in that batch.

    Inputs in `batch`:
      {
        "mel": (T_i, n_mels),
        "f0": (T_i,),
        "spk_id": scalar tensor,
        "length": scalar tensor (T_i),
      }

    Outputs:
      {
        "mel":      (B, T_max, n_mels),
        "f0":       (B, T_max),
        "spk_id":   (B,),
        "lengths":  (B,),
        "mask":     (B, T_max)  # 1 for valid frames, 0 for padded
      }
    """
    # Sort by length (optional but often useful)
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)

    mels = [item["mel"] for item in batch]
    f0s = [item["f0"] for item in batch]
    spk_ids = torch.stack([item["spk_id"] for item in batch], dim=0)  # (B,)
    lengths = torch.stack([item["length"] for item in batch], dim=0)  # (B,)

    B = len(batch)
    T_max = int(lengths.max().item())
    n_mels = mels[0].shape[1]

    mel_padded = torch.zeros(B, T_max, n_mels, dtype=torch.float32)
    f0_padded = torch.zeros(B, T_max, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.float32)

    for i in range(B):
        T_i = mels[i].shape[0]
        mel_padded[i, :T_i, :] = mels[i]
        f0_padded[i, :T_i] = f0s[i]
        mask[i, :T_i] = 1.0

    return {
        "mel": mel_padded,
        "f0": f0_padded,
        "spk_id": spk_ids,
        "lengths": lengths,
        "mask": mask,
    }
