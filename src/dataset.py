import os
import sys
import numpy as np
import soundfile as sf
from stft import stft
from features import features, targets


def build(clean_dir, noisy_dir, out_file, max_files=None):
    names = sorted(f for f in os.listdir(noisy_dir) if f.endswith(".wav"))
    if max_files:
        names = names[:max_files]
    F, G, V = [], [], []
    for i, name in enumerate(names):
        clean, _ = sf.read(os.path.join(clean_dir, name))
        noisy, _ = sf.read(os.path.join(noisy_dir, name))
        S, X = stft(clean), stft(noisy)
        g, v = targets(S, X)
        F.append(features(X))
        G.append(g)
        V.append(v)
        if (i + 1) % 100 == 0 or i + 1 == len(names):
            print(f"{i + 1}/{len(names)} files")
    F = np.concatenate(F).astype(np.float32)
    G = np.concatenate(G).astype(np.float32)
    V = np.concatenate(V).astype(np.float32)
    np.savez(out_file, features=F, gains=G, vad=V)
    print(f"saved {out_file}: {len(F)} frames ({len(F) / 100 / 60:.1f} minutes of audio)")


if __name__ == "__main__":
    clean_dir, noisy_dir, out_file = sys.argv[1:4]
    max_files = int(sys.argv[4]) if len(sys.argv) > 4 else None
    build(clean_dir, noisy_dir, out_file, max_files)