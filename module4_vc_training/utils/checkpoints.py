import os
import time
from typing import Dict, Any, Optional

import torch


def create_run_dir(out_dir: str, run_name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    best_val_loss: float,
    cfg: Dict[str, Any],
) -> None:
    payload = {
        "step": step,
        "epoch": epoch,
        "model": model_state,
        "optimizer": optimizer_state,
        "best_val_loss": best_val_loss,
        "cfg": cfg,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads a checkpoint saved by save_checkpoint().

    NOTE: This uses torch.load() which relies on pickle under the hood.
    Only load checkpoints you created yourself / trust.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # We intentionally keep weights_only=False because our checkpoint includes optimizer/state metadata.
    # Only load trusted local files.
    return torch.load(path, map_location=map_location)
