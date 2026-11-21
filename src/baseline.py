import numpy as np
import soundfile as sf
from pystoi import stoi
from stft import FS, HOP, stft, istft


def spectral_subtraction(x, noise_seconds=0.25, floor=0.1):
    X = stft(x)
    n_noise = int(noise_seconds * FS / HOP)
    noise_power = np.mean(np.abs(X[:n_noise]) ** 2, axis=0)
    gain = np.sqrt(np.maximum(1 - noise_power / (np.abs(X) ** 2 + 1e-12), floor ** 2))
    return istft(X * gain, len(x))


def snr(clean, y):
    return 10 * np.log10(np.sum(clean ** 2) / np.sum((clean - y) ** 2))


if __name__ == "__main__":
    name = "p232_001"
    clean, fs = sf.read(f"data/clean_testset_wav/{name}.wav")
    noisy, _ = sf.read(f"data/noisy_testset_wav/{name}.wav")
    base = spectral_subtraction(noisy)
    sf.write(f"outputs/{name}_specsub.wav", base, FS)
    results = {
        "noisy (input)": noisy,
        "spectral subtraction": base,
        "official RNNoise": sf.read(f"outputs/{name}_rnnoise.wav")[0],
        "ours, ideal gains": sf.read(f"outputs/{name}_ideal.wav")[0],
    }
    print(f"{'method':24s} {'SNR (dB)':>9s} {'STOI':>6s}")
    for label, y in results.items():
        y = y[: len(clean)]
        print(f"{label:24s} {snr(clean, y):9.1f} {stoi(clean, y, fs):6.3f}")
    print(f"saved outputs/{name}_specsub.wav")