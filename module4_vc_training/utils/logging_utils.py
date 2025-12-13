import os
import logging
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def create_logger(run_dir: str) -> logging.Logger:
    os.makedirs(run_dir, exist_ok=True)

    logger = logging.getLogger("module4_vc_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(run_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    return logger


def create_tensorboard_writer(log_dir: str, run_name: str) -> Optional["SummaryWriter"]:
    if SummaryWriter is None:
        return None

    os.makedirs(log_dir, exist_ok=True)
    tb_path = os.path.join(log_dir, run_name)
    return SummaryWriter(tb_path)
