import os
import yaml
import torch
from torch.utils.data import DataLoader

from datasets.vc_dataset import VCDataset, vc_collate_fn
from models.content_encoder import ContentEncoderWrapper
from models.decoder_vc import VCDecoder
from models.speaker_embedding import SpeakerEmbedding
from trainers.trainer_vc import VCTrainer
from utils.logging_utils import create_logger
from utils.checkpoints import create_run_dir

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "train_vc_content.yaml"
    )
    config_path = os.path.abspath(config_path)
    cfg = load_config(config_path)

    # Set seed
    seed = cfg["device"].get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    use_cuda = cfg["device"].get("use_cuda", True) and torch.cuda.is_available()
    device = torch.device(f"cuda:{cfg['device'].get('gpu_id', 0)}" if use_cuda else "cpu")

    # Create output/log directories
    run_dir = create_run_dir(cfg["checkpoint"]["out_dir"], cfg["logging"]["run_name"])
    logger = create_logger(run_dir)

    logger.info(f"Using device: {device}")
    logger.info(f"Run dir: {run_dir}")

    # Datasets & loaders
    train_ds = VCDataset(cfg["data"]["train_list"], cfg["data"]["segment_frames"], is_train=True)
    val_ds   = VCDataset(cfg["data"]["val_list"],   cfg["data"]["segment_frames"], is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        collate_fn=vc_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        collate_fn=vc_collate_fn,
    )

    # Models
    content_enc = ContentEncoderWrapper(cfg["model"]).to(device)
    decoder     = VCDecoder(cfg["model"]).to(device)
    spk_emb     = SpeakerEmbedding(cfg["model"]).to(device)

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
