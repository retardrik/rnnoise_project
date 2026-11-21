import numpy as np
import soundfile as sf
from stft import FS, FRAME, stft, istft

EBAND = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100]) * 4
NB = len(EBAND)
NBINS = FRAME // 2 + 1


def band_weights():
    W = np.zeros((NB, NBINS))
    k = np.arange(NBINS)
    for b in range(NB):
        if b > 0:
            lo, mid = EBAND[b - 1], EBAND[b]
            up = (k >= lo) & (k <= mid)
            W[b, up] = (k[up] - lo) / (mid - lo)
        if b < NB - 1:
            mid, hi = EBAND[b], EBAND[b + 1]
            down = (k >= mid) & (k <= hi)
            W[b, down] = (hi - k[down]) / (hi - mid)
    return W


W = band_weights()


def band_energy(X):
    return (np.abs(X) ** 2) @ W.T


def interp_gains(g):
    return g @ W


if __name__ == "__main__":
    clean, fs = sf.read("data/clean_testset_wav/p232_001.wav")
    noisy, _ = sf.read("data/noisy_testset_wav/p232_001.wav")
    S, X = stft(clean), stft(noisy)
    Es, Ex = band_energy(S), band_energy(X)
    g = np.minimum(np.sqrt(Es / (Ex + 1e-12)), 1)
    y = istft(X * interp_gains(g), len(noisy))
    total = W.sum(axis=0)[: EBAND[-1] + 1]
    print("bands:", NB, " band edges (Hz):", (EBAND * FS // FRAME).tolist())
    print("weights per bin add to:", total.min(), "to", total.max())
    print("band energy shape (frames, bands):", Ex.shape)
    sf.write("outputs/p232_001_ideal.wav", y, FS)
    print("saved outputs/p232_001_ideal.wav")