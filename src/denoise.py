import sys
import numpy as np
import soundfile as sf
import torch
from stft import FS, HOP, stft, istft
from bands import interp_gains
from features import features
from model import RNNoise

GAIN_FLOOR = 0.1
SPEECH_LEVEL_DB = -20.0


def load_model(model_file):
    model = RNNoise()
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()
    return model


def speech_loudness(y, vad, target_db=SPEECH_LEVEL_DB):
    is_speech = np.repeat(vad, HOP)[: len(y)] > 0.5
    if is_speech.sum() < FS // 10:
        return y
    rms = np.sqrt(np.mean(y[is_speech] ** 2))
    gain = 10 ** (target_db / 20) / rms
    gain = min(gain, 0.99 / (np.abs(y).max() + 1e-12))
    return y * gain


def denoise(noisy, model):
    X = stft(noisy)
    F = torch.tensor(features(X), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        gains, vad = model(F)
    gains = np.maximum(gains[0].numpy(), GAIN_FLOOR)
    smooth = np.zeros_like(gains)
    prev = np.zeros(gains.shape[1])
    for t in range(len(gains)):
        prev = np.maximum(0.6 * prev, gains[t])
        smooth[t] = prev
    y = istft(X * interp_gains(smooth), len(noisy))
    return speech_loudness(y, vad[0].numpy())


if __name__ == "__main__":
    model_file, in_file, out_file = sys.argv[1:4]
    noisy, fs = sf.read(in_file)
    y = denoise(noisy, load_model(model_file))
    sf.write(out_file, y, FS)
    print(f"{in_file} -> {out_file}")