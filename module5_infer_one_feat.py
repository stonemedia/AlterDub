import numpy as np
import torch
from module4_vc_training.models.content_encoder import ContentEncoderWrapper
from module4_vc_training.models.decoder_vc import VCDecoder
from module4_vc_training.models.speaker_embedding import SpeakerEmbedding

CKPT = "/workspace/altd_runs/checkpoints/rt_candidate_v1_best.pt"
MEL = "/workspace/AlterDub/module3_dataset/data/features/spk_f1/mel/spk_f1_000221.npy"
F0  = "/workspace/AlterDub/module3_dataset/data/features/spk_f1/pitch/spk_f1_000221.npy"
SPK_ID = 0
OUT = "/workspace/altd_runs/m5_mel_hat_spk_f1_000221.npy"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT, map_location="cpu")
    cfg = ckpt["cfg"]
    sd = ckpt["model"]

    content_enc = ContentEncoderWrapper(cfg["model"]).to(device).eval()
    decoder = VCDecoder(cfg["model"]).to(device).eval()
    spk_emb = SpeakerEmbedding(cfg["model"]).to(device).eval()

    content_enc.load_state_dict(sd["content_encoder"])
    decoder.load_state_dict(sd["decoder"])
    spk_emb.load_state_dict(sd["speaker_embedding"])

    mel = np.load(MEL)            # (80, T)
    f0  = np.load(F0)             # (T,)
    mel = torch.from_numpy(mel.T).unsqueeze(0).float().to(device)
    f0  = torch.from_numpy(f0).unsqueeze(0).float().to(device)
    spk_id = torch.tensor([SPK_ID], dtype=torch.long, device=device)

    with torch.no_grad():
        content = content_enc(mel)
        spk = spk_emb(spk_id)
        mel_hat = decoder(content, f0, spk)

    np.save(OUT, mel_hat.squeeze(0).cpu().numpy())
    print("Saved:", OUT, mel_hat.shape)

if __name__ == "__main__":
    main()
