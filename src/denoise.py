import sys
import numpy as np
import soundfile as sf
import torch
from stft import FS, stft, istft
from bands import interp_gains
from features import features
from model import RNNoise


def load_model(model_file):
    model = RNNoise()
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()
    return model


def denoise(noisy, model):
    X = stft(noisy)
    F = torch.tensor(features(X), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        gains, _ = model(F)
    gains = gains[0].numpy()
    smooth = np.zeros_like(gains)
    prev = np.zeros(gains.shape[1])
    for t in range(len(gains)):
        prev = np.maximum(0.6 * prev, gains[t])
        smooth[t] = prev
    return istft(X * interp_gains(smooth), len(noisy))


if __name__ == "__main__":
    model_file, in_file, out_file = sys.argv[1:4]
    noisy, fs = sf.read(in_file)
    y = denoise(noisy, load_model(model_file))
    sf.write(out_file, y, FS)
    print(f"{in_file} -> {out_file}")