import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

from module4_vc_training.datasets.vc_dataset import VCDataset, vc_collate_fn
from module4_vc_training.models.content_encoder import ContentEncoderWrapper
from module4_vc_training.models.decoder_vc import VCDecoder
from module4_vc_training.models.speaker_embedding import SpeakerEmbedding
from module4_vc_training.trainers.trainer_vc import VCTrainer
from module4_vc_training.utils.logging_utils import create_logger
from module4_vc_training.utils.checkpoints import create_run_dir


def load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args = parser.parse_args()

    default_cfg = os.path.abspath(os.path.join(script_dir, "..", "configs", "train_vc_content.yaml"))
    cfg_path = os.path.abspath(args.config) if args.config else default_cfg

    cfg = load_config(cfg_path)


    # Device
    use_cuda = cfg["device"].get("use_cuda", True) and torch.cuda.is_available()
    gpu_id = int(cfg["device"].get("gpu_id", 0))
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")

    # Run dir
    run_dir = create_run_dir(cfg["checkpoint"]["out_dir"], cfg["logging"]["run_name"])
    logger = create_logger(run_dir)

    logger.info(f"Using device: {device}")
    logger.info(f"Config: {cfg_path}")

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

    trainer.train()


if __name__ == "__main__":
    main()
