import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FS = 48000
FRAME = 480
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEMO = os.path.join(HERE, "rnnoise_demo.exe" if os.name == "nt" else "rnnoise_demo")


def load_48k_mono(path):
    x, fs = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if fs != FS:
        g = np.gcd(fs, FS)
        x = resample_poly(x, FS // g, fs // g)
    return x


def denoise(in_wav, out_wav, demo_path=DEFAULT_DEMO):
    x = load_48k_mono(in_wav)
    n = len(x)
    x = np.concatenate([x, np.zeros(FRAME)])
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with tempfile.TemporaryDirectory() as d:
        raw_in = os.path.join(d, "in.raw")
        raw_out = os.path.join(d, "out.raw")
        pcm.tofile(raw_in)
        subprocess.run([demo_path, raw_in, raw_out], check=True)
        y = np.fromfile(raw_out, dtype="<i2").astype("float64") / 32768
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    sf.write(out_wav, y, FS, subtype="PCM_16")
    print(f"{in_wav} -> {out_wav}  ({len(y) / FS:.2f} s at 48 kHz)")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("usage: python run_rnnoise.py noisy.wav denoised.wav [path/to/rnnoise_demo]")
        sys.exit(1)
    denoise(*sys.argv[1:])
