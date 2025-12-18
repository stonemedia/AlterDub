import torch
from module4_vc_training.models.content_encoder import ContentEncoderWrapper
from module4_vc_training.models.decoder_vc import VCDecoder
from module4_vc_training.models.speaker_embedding import SpeakerEmbedding

CKPT_PATH = "/workspace/altd_runs/checkpoints/rt_candidate_v1_best.pt"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg = ckpt["cfg"]
    sd = ckpt["model"]

    content_enc = ContentEncoderWrapper(cfg["model"]).to(device).eval()
    decoder = VCDecoder(cfg["model"]).to(device).eval()
    spk_emb = SpeakerEmbedding(cfg["model"]).to(device).eval()

    content_enc.load_state_dict(sd["content_encoder"])
    decoder.load_state_dict(sd["decoder"])
    spk_emb.load_state_dict(sd["speaker_embedding"])

    B, T, n_mels = 2, 128, int(cfg["model"]["n_mels"])
    mel = torch.randn(B, T, n_mels, device=device)
    f0  = torch.randn(B, T, device=device)
    spk_id = torch.tensor([0, 1], dtype=torch.long, device=device)

    with torch.no_grad():
        content = content_enc(mel)
        spk = spk_emb(spk_id)
        mel_hat = decoder(content, f0, spk)

    print("device:", device)
  

    print("content:", tuple(content.shape))
    print("spk:", tuple(spk.shape))
    print("mel_hat:", tuple(mel_hat.shape))

if __name__ == "__main__":
    main()
