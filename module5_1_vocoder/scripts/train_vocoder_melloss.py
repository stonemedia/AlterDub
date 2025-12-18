import os
import time
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from module5_1_vocoder.models.dataset import VocoderPairDataset
from module5_1_vocoder.models.hifigan_generator import Generator
from module5_1_vocoder.models.hifigan_discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
from module5_1_vocoder.models.losses import discriminator_loss, generator_loss, feature_loss
from module5_1_vocoder.models.mel import MelSTFT


def save_ckpt(path, step, G, mpd, msd, opt_g, opt_d, scaler):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'step': step,
        'G': G.state_dict(),
        'mpd': mpd.state_dict(),
        'msd': msd.state_dict(),
        'opt_g': opt_g.state_dict(),
        'opt_d': opt_d.state_dict(),
        'scaler': scaler.state_dict(),
    }, path)


def load_ckpt(path, G, mpd, msd, opt_g, opt_d, scaler, device):
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt['G'])
    mpd.load_state_dict(ckpt['mpd'])
    msd.load_state_dict(ckpt['msd'])
    opt_g.load_state_dict(ckpt['opt_g'])
    opt_d.load_state_dict(ckpt['opt_d'])
    if 'scaler' in ckpt and ckpt['scaler'] is not None:
        scaler.load_state_dict(ckpt['scaler'])
    return int(ckpt.get('step', 0))


def main(cfg_path: str):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Device: {device}")

    ds = VocoderPairDataset(
        cfg['filelist'],
        segment_mel_frames=int(cfg['segment_mel_frames']),
        sample_rate=int(cfg['sample_rate']),
        hop_length=int(cfg['hop_length']),
        n_mels=int(cfg['n_mels']),
        shuffle=True,
    )
    dl = DataLoader(
        ds,
        batch_size=int(cfg['batch_size']),
        shuffle=True,
        num_workers=int(cfg['num_workers']),
        pin_memory=True,
        drop_last=True,
    )

    G = Generator(n_mels=int(cfg['n_mels'])).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    mel_fn = MelSTFT(
        sample_rate=int(cfg['sample_rate']),
        n_fft=int(cfg.get('n_fft', 512)),
        win_length=int(cfg.get('win_length', 160)),
        hop_length=int(cfg['hop_length']),
        n_mels=int(cfg['n_mels']),
        fmin=float(cfg.get('f0min', 0.0)),
        fmax=float(cfg.get('f0max', 8000.0)),
    ).to(device)

    lr = float(cfg['lr'])
    b0, b1 = cfg['betas']
    opt_g = torch.optim.AdamW(G.parameters(), lr=lr, betas=(float(b0), float(b1)), weight_decay=float(cfg.get('weight_decay', 0.0)))
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(msd.parameters()), lr=lr, betas=(float(b0), float(b1)), weight_decay=float(cfg.get('weight_decay', 0.0)))

    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

    ckpt_dir = cfg['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    run_name = cfg.get('run_name', 'vocoder_run') + '_melloss'
    latest_ckpt = os.path.join(ckpt_dir, f"{run_name}_latest.pt")

    step = 0
    if os.path.exists(latest_ckpt):
        print(f"[INFO] Resuming from: {latest_ckpt}")
        step = load_ckpt(latest_ckpt, G, mpd, msd, opt_g, opt_d, scaler, device)
        print(f"[INFO] Resumed at step={step}")

    lambda_fm = float(cfg.get('lambda_fm', 10.0))
    lambda_mel = float(cfg.get('lambda_mel', 45.0))

    # 🔒 Stability clamps for log-mel(dB)
    MEL_DB_MIN = -11.5
    MEL_DB_MAX = 2.0
    max_steps = int(cfg.get('max_steps', 200000))
    log_every = int(cfg.get('log_every', 50))
    save_every = int(cfg.get('save_every', 2000))

    G.train(); mpd.train(); msd.train(); mel_fn.train()
    t0 = time.time()

    while step < max_steps:
        for mel_target, y in dl:
            if step >= max_steps:
                break

            mel_target = mel_target.to(device, non_blocking=True)       # (B,80,T)
            y = y.to(device, non_blocking=True).unsqueeze(1)            # (B,1,T_samples)

            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                y_hat = G(mel_target)

            # ----- D -----
            opt_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                y_df_r, y_df_g, _, _ = mpd(y, y_hat.detach())
                y_ds_r, y_ds_g, _, _ = msd(y, y_hat.detach())
                loss_d_mpd, _ = discriminator_loss(y_df_r, y_df_g)
                loss_d_msd, _ = discriminator_loss(y_ds_r, y_ds_g)
                loss_d = loss_d_mpd + loss_d_msd
            scaler.scale(loss_d).backward()
            torch.nn.utils.clip_grad_norm_(list(mpd.parameters())+list(msd.parameters()), 1.0)
            scaler.step(opt_d)

            # ----- G -----
            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                y_df_r, y_df_g, fmap_f_r, fmap_f_g = mpd(y, y_hat)
                y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = msd(y, y_hat)

                loss_g_adv_mpd, _ = generator_loss(y_df_g)
                loss_g_adv_msd, _ = generator_loss(y_ds_g)
                loss_g_adv = loss_g_adv_mpd + loss_g_adv_msd

                loss_fm = feature_loss(fmap_f_r, fmap_f_g) + feature_loss(fmap_s_r, fmap_s_g)

                mel_pred = mel_fn(y_hat)
                mel_pred = mel_pred[:, :, :mel_target.size(2)]  # 🔒 align frames

                # 🔒 clamp both to avoid inf/NaN explosions in log-mel(dB)
                mel_pred_c = torch.clamp(mel_pred, min=MEL_DB_MIN, max=MEL_DB_MAX)
                mel_tgt_c  = torch.clamp(mel_target, min=MEL_DB_MIN, max=MEL_DB_MAX)

                # more stable than pure L1 at start
                loss_mel = F.smooth_l1_loss(mel_pred_c, mel_tgt_c)

                loss_g = loss_g_adv + lambda_fm * loss_fm + lambda_mel * loss_mel

                if (not torch.isfinite(loss_g)) or (not torch.isfinite(loss_d)):
                    print(f"[FATAL] non-finite loss at step={step}: loss_d={loss_d.item()} loss_g={loss_g.item()}")
                    return

            scaler.scale(loss_g).backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler.step(opt_g)
            scaler.update()

            step += 1

            if step % log_every == 0:
                dt = time.time() - t0
                print(f"[STEP {step}] loss_d={loss_d.item():.4f} loss_g={loss_g.item():.4f} adv={loss_g_adv.item():.4f} fm={loss_fm.item():.4f} mel={loss_mel.item():.4f} ({dt:.1f}s)")
                t0 = time.time()

            if step % save_every == 0:
                save_ckpt(latest_ckpt, step, G, mpd, msd, opt_g, opt_d, scaler)
                tagged = os.path.join(ckpt_dir, f"{run_name}_step{step}.pt")
                save_ckpt(tagged, step, G, mpd, msd, opt_g, opt_d, scaler)
                print(f"[CKPT] Saved: {tagged}")

    save_ckpt(latest_ckpt, step, G, mpd, msd, opt_g, opt_d, scaler)
    print(f"[DONE] Training finished at step={step}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    args = ap.parse_args()
    main(args.config)

