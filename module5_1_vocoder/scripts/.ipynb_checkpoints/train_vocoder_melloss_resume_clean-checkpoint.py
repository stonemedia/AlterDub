import os
import time
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.utils as nn_utils

from module5_1_vocoder.models.dataset import VocoderPairDataset
from module5_1_vocoder.models.hifigan_generator import Generator
from module5_1_vocoder.models.hifigan_discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
from module5_1_vocoder.models.losses import discriminator_loss, generator_loss, feature_loss
from module5_1_vocoder.models.mel import MelSTFT
from module5_1_vocoder.models.mrstft_loss import MRSTFTLoss


def save_ckpt(path, step, G, mpd, msd, opt_g, opt_d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": int(step),
        "G": G.state_dict(),
        "mpd": mpd.state_dict(),
        "msd": msd.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        # AMP is intentionally not used here (pure fp32 stability).
        "scaler": None,
    }, path)


def load_ckpt(path, G, mpd, msd, opt_g, opt_d, device, load_optim=True):
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G"])
    mpd.load_state_dict(ckpt["mpd"])
    msd.load_state_dict(ckpt["msd"])
    if load_optim:
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
    return int(ckpt.get("step", 0))

def main(cfg_path: str, resume_ckpt: str = "", reset_optim: bool = False):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # Dataset / loader
    ds = VocoderPairDataset(
        cfg["filelist"],
        segment_mel_frames=int(cfg["segment_mel_frames"]),
        sample_rate=int(cfg["sample_rate"]),
        hop_length=int(cfg["hop_length"]),
        n_mels=int(cfg["n_mels"]),
        shuffle=True,
    )
    dl = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )


    # Models
    G = Generator(n_mels=int(cfg["n_mels"])).to(device).float()
    mpd = MultiPeriodDiscriminator().to(device).float()
    msd = MultiScaleDiscriminator().to(device).float()

    # Mel extractor (same as your current file)
    mel_fn = MelSTFT(
        sample_rate=int(cfg["sample_rate"]),
        n_fft=int(cfg.get("n_fft", 512)),
        win_length=int(cfg.get("win_length", 160)),
        hop_length=int(cfg["hop_length"]),
        n_mels=int(cfg["n_mels"]),
        fmin=float(cfg.get("f0min", 0.0)),
        fmax=float(cfg.get("f0max", 8000.0)),
    ).to(device)

    mrstft = MRSTFTLoss().to(device)

    lr = float(cfg["lr"])
    b0, b1 = cfg["betas"]
    opt_g = torch.optim.AdamW(
        G.parameters(),
        lr=lr,
        betas=(float(b0), float(b1)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    opt_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=lr,
        betas=(float(b0), float(b1)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    ckpt_dir = cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    run_name = cfg.get("run_name", "vocoder_run") + "_melloss"
    latest_ckpt = os.path.join(ckpt_dir, f"{run_name}_latest.pt")

    step = 0
    if resume_ckpt:
        print(f"[INFO] Resuming from explicit ckpt: {resume_ckpt}")
        step = load_ckpt(resume_ckpt, G, mpd, msd, opt_g, opt_d, device, load_optim=(not reset_optim))
        print(f"[INFO] Resumed at step={step} (reset_optim={reset_optim})")
    elif os.path.exists(latest_ckpt):
        print(f"[INFO] Resuming from latest: {latest_ckpt}")
        step = load_ckpt(latest_ckpt, G, mpd, msd, opt_g, opt_d, device, load_optim=(not reset_optim))
        print(f"[INFO] Resumed at step={step} (reset_optim={reset_optim})")


    lambda_fm  = float(cfg.get("lambda_fm", 10.0))
    lambda_mel = float(cfg.get("lambda_mel", 10.0))
    lambda_stft = float(cfg.get("lambda_stft", 1.0))

    # Keep exactly as in your current script (even if odd); we can tune later.
    MEL_DB_MIN = -11.5
    MEL_DB_MAX = 2.0

    max_steps = int(cfg.get("max_steps", 200000))
    log_every  = int(cfg.get("log_every", 50))
    save_every = int(cfg.get("save_every", 1000))

    grad_clip_g = float(cfg.get("grad_clip_g", 1.0))
    grad_clip_d = float(cfg.get("grad_clip_d", 1.0))
    G.train(); mpd.train(); msd.train(); mel_fn.train()
    t0 = time.time()

    while step < max_steps:
        for mel_target, y in dl:
            if step >= max_steps:
                break

            mel_target = mel_target.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).unsqueeze(1).float()

            # ---------
            # Forward G
            # ---------
            y_hat = G(mel_target)

            # -------------------------
            # Discriminator (FP32 only)
            # -------------------------
            opt_d.zero_grad(set_to_none=True)
            y_hat_det = y_hat.detach()

            y_df_r, y_df_g, _, _ = mpd(y, y_hat_det)
            y_ds_r, y_ds_g, _, _ = msd(y, y_hat_det)
            loss_d_mpd, _ = discriminator_loss(y_df_r, y_df_g)
            loss_d_msd, _ = discriminator_loss(y_ds_r, y_ds_g)
            loss_d = loss_d_mpd + loss_d_msd

            if not torch.isfinite(loss_d):
                print(f"[WARN] non-finite loss_d at step={step}: {loss_d}. Skipping batch.")
                step += 1
                continue


            loss_d.backward()
            nn_utils.clip_grad_norm_(list(mpd.parameters()) + list(msd.parameters()), grad_clip_d)
            opt_d.step()

            # -------------------------
            # Generator (FP32 only)
            # -------------------------
            opt_g.zero_grad(set_to_none=True)

            y_df_r, y_df_g, fmap_f_r, fmap_f_g = mpd(y, y_hat)
            y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = msd(y, y_hat)


            loss_g_adv_mpd, _ = generator_loss(y_df_g)
            loss_g_adv_msd, _ = generator_loss(y_ds_g)
            loss_g_adv = loss_g_adv_mpd + loss_g_adv_msd
            loss_fm = feature_loss(fmap_f_r, fmap_f_g) + feature_loss(fmap_s_r, fmap_s_g)

            # ---- mel_pred block (exactly as your current file) ----
            mel_pred = mel_fn(y_hat)
            mel_pred = mel_pred[:, :, :mel_target.size(2)]
            mel_pred_c = torch.clamp(mel_pred, min=MEL_DB_MIN, max=MEL_DB_MAX)
            mel_tgt_c  = torch.clamp(mel_target, min=MEL_DB_MIN, max=MEL_DB_MAX)
            loss_mel = F.smooth_l1_loss(mel_pred_c, mel_tgt_c)
            # ------------------------------------------------------

            # MR-STFT loss (reduces wobble/shiver, improves transients)
            loss_stft = mrstft(y_hat, y)

            loss_g = loss_g_adv + lambda_fm * loss_fm + lambda_mel * loss_mel + lambda_stft * loss_stft

            if not torch.isfinite(loss_g):
                print(f"[WARN] non-finite loss_g at step={step}: {loss_g}. Skipping batch.")
                step += 1
                continue

            loss_g.backward()
            nn_utils.clip_grad_norm_(G.parameters(), grad_clip_g)
            opt_g.step()

            step += 1


            if step % log_every == 0:
                dt = time.time() - t0
                print(
                    f"[STEP {step}] loss_d={loss_d.item():.4f} "
                    f"loss_g={loss_g.item():.4f} adv={loss_g_adv.item():.4f} "
                    f"fm={loss_fm.item():.4f} mel={loss_mel.item():.4f} stft={loss_stft.item():.4f} ({dt:.1f}s)"
                )
                t0 = time.time()

            if step % save_every == 0:
                save_ckpt(latest_ckpt, step, G, mpd, msd, opt_g, opt_d)
                tagged = os.path.join(ckpt_dir, f"{run_name}_step{step}.pt")
                save_ckpt(tagged, step, G, mpd, msd, opt_g, opt_d)
                print(f"[CKPT] Saved: {tagged}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume_ckpt", type=str, default="")
    ap.add_argument("--reset_optim", action="store_true")
    args = ap.parse_args()
    main(args.config, args.resume_ckpt, reset_optim=args.reset_optim)
