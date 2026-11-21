import numpy as np
import soundfile as sf

FS = 48000
FRAME = 960
HOP = 480


def vorbis_window(n=FRAME):
    i = np.arange(n)
    return np.sin(0.5 * np.pi * np.sin(np.pi * i / n) ** 2)


def stft(x):
    x = np.concatenate([np.zeros(HOP), x, np.zeros(FRAME)])
    n_frames = (len(x) - FRAME) // HOP + 1
    w = vorbis_window()
    frames = np.stack([x[m * HOP : m * HOP + FRAME] * w for m in range(n_frames)])
    return np.fft.rfft(frames, axis=1)


def istft(X, length):
    w = vorbis_window()
    frames = np.fft.irfft(X, n=FRAME, axis=1) * w
    y = np.zeros((len(frames) - 1) * HOP + FRAME)
    for m in range(len(frames)):
        y[m * HOP : m * HOP + FRAME] += frames[m]
    return y[HOP : HOP + length]


if __name__ == "__main__":
    x, fs = sf.read("data/noisy_testset_wav/p232_001.wav")
    X = stft(x)
    y = istft(X, len(x))
    print("STFT shape (frames, bins):", X.shape)
    print("reconstruction error:", np.abs(x - y).max())
    sf.write("outputs/p232_001_roundtrip.wav", y, FS)
    print("saved outputs/p232_001_roundtrip.wav")