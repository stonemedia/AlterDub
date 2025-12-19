import torch
import torch.nn as nn
import torch.nn.functional as F

def stft_mag(x, n_fft, hop, win):
    # x: [B, 1, T] or [B, T]
    if x.dim() == 3:
        x = x.squeeze(1)
    window = torch.hann_window(win, device=x.device, dtype=x.dtype)
    X = torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, center=True, return_complex=True
    )
    mag = torch.abs(X)  # [B, F, TT]
    return mag
class MRSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss:
      - Spectral convergence
      - Log magnitude L1
    """
    def __init__(self, resolutions=None, eps=1e-7):
        super().__init__()
        if resolutions is None:
            # Good “audio vocoder” defaults
            resolutions = [
                (1024, 120, 600),
                (2048, 240, 1200),
                (512,  50, 240),
            ]
        self.resolutions = resolutions
        self.eps = eps
    def forward(self, y_hat, y):
        sc_loss = 0.0
        mag_loss = 0.0

        for (n_fft, hop, win) in self.resolutions:
            y_mag  = stft_mag(y,     n_fft, hop, win)
            yh_mag = stft_mag(y_hat, n_fft, hop, win)

            # Spectral convergence: ||Y - YH|| / ||Y||
            diff = torch.norm(y_mag - yh_mag, p='fro')
            denom = torch.norm(y_mag, p='fro').clamp(min=self.eps)
            sc = diff / denom

            # Log magnitude L1
            log_y  = torch.log(y_mag.clamp(min=self.eps))
            log_yh = torch.log(yh_mag.clamp(min=self.eps))
            mag = F.l1_loss(log_yh, log_y)

            sc_loss += sc
            mag_loss += mag

        sc_loss = sc_loss / len(self.resolutions)
        mag_loss = mag_loss / len(self.resolutions)
        return sc_loss + mag_loss
