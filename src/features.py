import numpy as np
import soundfile as sf
from scipy.fft import dct
from stft import stft
from bands import band_energy


def features(X):
    E = band_energy(X)
    bfcc = dct(np.log10(1e-2 + E), type=2, norm="ortho", axis=1)
    c = bfcc[:, :6]
    prev1 = np.vstack([c[:1], c[:-1]])
    prev2 = np.vstack([c[:1], c[:1], c[:-2]])
    d1 = c - prev1
    d2 = c - 2 * prev1 + prev2
    return np.hstack([bfcc, d1, d2])


def targets(S, X):
    Es, Ex = band_energy(S), band_energy(X)
    g = np.minimum(np.sqrt(Es / (Ex + 1e-12)), 1)
    frame_energy = Es.sum(axis=1)
    vad = (frame_energy > 1e-3 * frame_energy.max()).astype(float)
    return g, vad


if __name__ == "__main__":
    clean, _ = sf.read("data/clean_testset_wav/p232_001.wav")
    noisy, _ = sf.read("data/noisy_testset_wav/p232_001.wav")
    S, X = stft(clean), stft(noisy)
    F = features(X)
    g, vad = targets(S, X)
    print("features (frames, numbers per frame):", F.shape)
    print("target gains (frames, bands):", g.shape, " range:", round(g.min(), 3), "to", round(g.max(), 3))
    print("speech frames:", int(vad.sum()), "of", len(vad))
    print("frame 100, first 5 features:", np.round(F[100, :5], 2))