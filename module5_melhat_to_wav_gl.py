import numpy as np
import librosa
import soundfile as sf

MELHAT = "/workspace/altd_runs/m5_mel_hat_spk_f1_000221.npy"
OUTWAV = "/workspace/altd_runs/m5_out_spk_f1_000221_gl.wav"

# Must match module2_feature_extractor
SR = 16000
N_FFT = 512
WIN_LENGTH = 160   # 0.010*SR
HOP_LENGTH = 80    # 0.005*SR
N_MELS = 80
FMIN = 0
FMAX = SR // 2

def main():
    mel = np.load(MELHAT)  # (T, 80)
    if mel.ndim != 2 or mel.shape[1] != N_MELS:
        raise ValueError(f"Expected (T,{N_MELS}), got {mel.shape}")

    # Convert mel (log-mel assumed) back to linear spectrogram
    mel_T = mel.T  # (80, T) for librosa
    S = librosa.feature.inverse.mel_to_stft(
        mel_T,
        sr=SR,
        n_fft=N_FFT,
        power=1.0,
        fmin=FMIN,
        fmax=FMAX
    )

    # Griffin–Lim phase reconstruction
    wav = librosa.griffinlim(
        S,
        n_iter=32,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH
    )

    sf.write(OUTWAV, wav, SR)
    print("WAV written:", OUTWAV, "length_sec:", len(wav)/SR)

if __name__ == "__main__":
    main()
