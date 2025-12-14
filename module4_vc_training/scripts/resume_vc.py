import os
import yaml
import torch
from torch.utils.data import DataLoader

from module4_vc_training.datasets.vc_dataset import VCDataset, vc_collate_fn
from module4_vc_training.models.content_encoder import ContentEncoderWrapper
from module4_vc_training.models.decoder_vc import VCDecoder
from module4_vc_training.models.speaker_embedding import SpeakerEmbedding
from module4_vc_training.trainers.trainer_vc import VCTrainer
from module4_vc_training.utils.logging_utils import create_logger
from module4_vc_training.utils.checkpoints import load_checkpoint


def load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # ---- Edit these two lines when resuming ----
    CKPT_PATH = r"/workspace/altd_runs/checkpoints/vc_content_encoder_gpu_20251214_132942/last.pt"
    CFG_PATH = r"/workspace/AlterDub/module4_vc_training/configs/train_vc_content_gpu.yaml"
    # ------------------------------------------

    cfg = load_config(CFG_PATH)

    # Device
    use_cuda = cfg["device"].get("use_cuda", True) and torch.cuda.is_available()
    gpu_id = int(cfg["device"].get("gpu_id", 0))
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")

    # Load checkpoint
    ckpt = load_checkpoint(CKPT_PATH, map_location=str(device))

    # Resume into same run_dir (folder where checkpoint lives)
    run_dir = os.path.dirname(CKPT_PATH)
    logger = create_logger(run_dir)
    logger.info(f"Resuming from checkpoint: {CKPT_PATH}")
    logger.info(f"Using device: {device}")

    # Datasets
    train_list = cfg["data"]["train_list"]
    val_list = cfg["data"]["val_list"]
    segment_frames = int(cfg["data"]["segment_frames"])

    train_ds = VCDataset(train_list, segment_frames=segment_frames, is_train=True)
    val_ds = VCDataset(val_list, segment_frames=segment_frames, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        collate_fn=vc_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        collate_fn=vc_collate_fn,
    )

    # Models
    content_enc = ContentEncoderWrapper(cfg["model"]).to(device)
    decoder = VCDecoder(cfg["model"]).to(device)
    spk_emb = SpeakerEmbedding(cfg["model"]).to(device)

    # Load model weights
    content_enc.load_state_dict(ckpt["model"]["content_encoder"])
    decoder.load_state_dict(ckpt["model"]["decoder"])
    spk_emb.load_state_dict(ckpt["model"]["speaker_embedding"])

    # Trainer
    trainer = VCTrainer(
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        logger=logger,
        content_encoder=content_enc,
        decoder=decoder,
        speaker_embedding=spk_emb,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Load optimizer + step/epoch counters
    trainer.optimizer.load_state_dict(ckpt["optimizer"])
    trainer.step = int(ckpt.get("step", 0))
    trainer.epoch = int(ckpt.get("epoch", 0))
    trainer.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    logger.info(f"Resumed state: step={trainer.step}, epoch={trainer.epoch}, best_val_loss={trainer.best_val_loss}")

    trainer.train()


if __name__ == "__main__":
    main()
