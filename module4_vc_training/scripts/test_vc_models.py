import os
import torch
from torch.utils.data import DataLoader
import yaml

from module4_vc_training.datasets.vc_dataset import VCDataset, vc_collate_fn
from module4_vc_training.models.content_encoder import ContentEncoderWrapper
from module4_vc_training.models.decoder_vc import VCDecoder
from module4_vc_training.models.speaker_embedding import SpeakerEmbedding


def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.abspath(os.path.join(script_dir, "..", "configs", "train_vc_content.yaml"))
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def find_train_list() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

    candidates = [
        os.path.join(root_dir, "module3_dataset", "train_list.txt"),
        os.path.join(root_dir, "module3_dataset", "metadata", "train_list.txt"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            print(f"[INFO] Using train_list at: {path}")
            return path

    msg_lines = ["[ERROR] Could not find train_list.txt. Tried:"]
    for path in candidates:
        msg_lines.append(f"  - {path}")
    raise FileNotFoundError("\n".join(msg_lines))


def main():
    cfg = load_config()

    list_path = find_train_list()
    segment_frames = cfg["data"]["segment_frames"]

    ds = VCDataset(list_path=list_path, segment_frames=segment_frames, is_train=True)
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=vc_collate_fn,
    )

    batch = next(iter(loader))
    mel = batch["mel"]         # (B, T, n_mels)
    f0 = batch["f0"]           # (B, T)
    spk_id = batch["spk_id"]   # (B,)
    lengths = batch["lengths"] # (B,)
    mask = batch["mask"]       # (B, T)

    print("mel:", mel.shape)
    print("f0:", f0.shape)
    print("spk_id:", spk_id.shape)
    print("lengths:", lengths)
    print("mask:", mask.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build models
    content_enc = ContentEncoderWrapper(cfg["model"]).to(device)
    decoder = VCDecoder(cfg["model"]).to(device)
    spk_emb_module = SpeakerEmbedding(cfg["model"]).to(device)

    mel = mel.to(device)
    f0 = f0.to(device)
    spk_id = spk_id.to(device)
    lengths = lengths.to(device)
    mask = mask.to(device)

    # Forward pass
    content_repr = content_enc(mel, lengths=lengths, mask=mask)   # (B, T, content_dim)
    spk_emb = spk_emb_module(spk_id)                             # (B, spk_emb_dim)
    mel_hat = decoder(content_repr, f0, spk_emb, lengths=lengths, mask=mask)

    print("content_repr:", content_repr.shape)
    print("spk_emb:", spk_emb.shape)
    print("mel_hat:", mel_hat.shape)


if __name__ == "__main__":
    main()
