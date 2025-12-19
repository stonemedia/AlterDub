import torch
import torch.nn as nn
import torch.nn.functional as F

class MelSTFT(nn.Module):
    """
    Computes log-mel in dB with base-10:
        mel_db = 10 * log10(mel_power + eps)

    This matches AlterDub Module-2 locked definition.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 160,
        hop_length: int = 80,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        eps: float = 1e-10,
        clamp_db_min: float = -100.0,
        to_db10: bool = True,
        center: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps
        self.clamp_db_min = clamp_db_min
        self.to_db10 = to_db10
        self.center = center
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

        # Create mel filterbank (triangular) using torch (no librosa dependency)
        self.register_buffer("mel_fb", self._build_mel_filter())

    def _hz_to_mel(self, hz: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filter(self) -> torch.Tensor:
        # FFT freq bins
        n_freq = self.n_fft // 2 + 1
        device = self.window.device
        dtype = torch.float32
        fmin = torch.tensor(self.fmin, dtype=dtype, device=device)
        fmax = torch.tensor(self.fmax, dtype=dtype, device=device)

        mmin = self._hz_to_mel(fmin)
        mmax = self._hz_to_mel(fmax)

        m_pts = torch.linspace(mmin, mmax, self.n_mels + 2, device=device, dtype=dtype)
        f_pts = self._mel_to_hz(m_pts)

        # bin frequencies
        fft_freqs = torch.linspace(0, self.sample_rate / 2, n_freq, device=device, dtype=dtype)

        fb = torch.zeros((self.n_mels, n_freq), device=device, dtype=dtype)
        for i in range(self.n_mels):
            f_left, f_center, f_right = f_pts[i], f_pts[i+1], f_pts[i+2]
            # rising slope
            left_slope = (fft_freqs - f_left) / (f_center - f_left + 1e-20)
            # falling slope
            right_slope = (f_right - fft_freqs) / (f_right - f_center + 1e-20)
            fb[i] = torch.clamp(torch.minimum(left_slope, right_slope), min=0.0)

        # Normalize filterbank energy (optional but common)
        enorm = 2.0 / (f_pts[2:self.n_mels+2] - f_pts[:self.n_mels])
        fb *= enorm.unsqueeze(1)

        return fb
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B,1,T) float32 in [-1,1]
        returns mel: (B, n_mels, T_frames)
        """
        if y.dim() != 3:
            raise ValueError(f"Expected y (B,1,T), got {y.shape}")

        # STFT -> magnitude
        stft = torch.stft(
            y.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            return_complex=True,
        )  # (B, F, TT)
        mag = stft.abs()  # magnitude
        power = mag * mag  # power spectrum
        # mel power
        mel_fb = self.mel_fb
        if mel_fb.device != power.device:
            mel_fb = mel_fb.to(power.device)
        mel_power = torch.matmul(mel_fb, power)  # (n_mels, F) x (B,F,TT) -> (B,n_mels,TT)

        if self.to_db10:
            mel_db = 10.0 * torch.log10(torch.clamp(mel_power, min=self.eps))
            mel_db = torch.clamp(mel_db, min=self.clamp_db_min)
            return mel_db
        else:
            # fallback: natural log
            return torch.log(torch.clamp(mel_power, min=self.eps))
