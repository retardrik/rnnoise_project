"""Local demo server for the RNNoise re-implementation.

Double-click demo\\start.bat, or from the project folder run:
    venv\\Scripts\\python demo\\server.py
then open http://127.0.0.1:8765

Needs rnnoise_model.pt in the project folder. It only imports from src/
(nothing there is modified).
"""
import base64
import io
import json
import os
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pystoi import stoi  # noqa: E402

from baseline import snr, spectral_subtraction  # noqa: E402
from denoise import denoise, load_model  # noqa: E402
from stft import FS, stft  # noqa: E402

MODEL_FILE = os.path.join(PROJECT, "rnnoise_model.pt")
SAMPLE_FILE = os.path.join(PROJECT, "test_noisy.wav")
DATA = os.path.join(PROJECT, "data")
MAX_BYTES = 60 * 1024 * 1024
MAX_SECONDS = 60
PORT = 8765

print("loading", MODEL_FILE)
MODEL = load_model(MODEL_FILE)


# ---------- audio helpers ----------

def read_audio(raw):
    x, fs = sf.read(io.BytesIO(raw), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if fs != FS:
        g = np.gcd(int(fs), FS)
        x = resample_poly(x, FS // g, int(fs) // g)
    return x[: MAX_SECONDS * FS]


def make_noise(kind, n, rng):
    if kind == "white":
        return rng.standard_normal(n)
    if kind == "pink":
        spec = np.fft.rfft(rng.standard_normal(n))
        f = np.arange(len(spec))
        spec[1:] /= np.sqrt(f[1:])
        spec[0] = 0
        return np.fft.irfft(spec, n)
    if kind == "hum":
        t = np.arange(n) / FS
        hum = sum(np.sin(2 * np.pi * 50 * k * t) / k for k in range(1, 8))
        return hum + 0.3 * rng.standard_normal(n)
    if kind == "dataset":
        clean_dir = os.path.join(DATA, "clean_trainset_28spk_wav")
        noisy_dir = os.path.join(DATA, "noisy_trainset_28spk_wav")
        names = [f for f in os.listdir(noisy_dir) if f.endswith(".wav")]
        pieces, total = [], 0
        while total < n:
            name = rng_choice(rng, names)
            c, _ = sf.read(os.path.join(clean_dir, name))
            y, _ = sf.read(os.path.join(noisy_dir, name))
            pieces.append(y - c)
            total += len(c)
        return np.concatenate(pieces)[:n]
    raise ValueError(f"unknown noise type {kind}")


def rng_choice(rng, items):
    return items[int(rng.integers(len(items)))]


def mix_at_snr(clean, noise, snr_db):
    p_s = np.mean(clean ** 2) + 1e-12
    p_n = np.mean(noise ** 2) + 1e-12
    return clean + noise * np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))


def frame_levels_db(x):
    X = stft(x)
    return 10 * np.log10(np.sum(np.abs(X) ** 2, axis=1) + 1e-10)


def level_stats(x):
    """Rough, reference-free levels: noise floor = quiet frames, speech = loud frames."""
    e = frame_levels_db(x)
    floor, speech = np.percentile(e, 10), np.percentile(e, 95)
    return {"noise_floor_db": float(floor), "speech_db": float(speech),
            "est_snr_db": float(speech - floor)}


def to_wav_b64(x, scale):
    buf = io.BytesIO()
    sf.write(buf, np.clip(x * scale, -1, 1), FS, subtype="PCM_16", format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def spectrogram_b64(x, vmin, vmax):
    X = stft(x)
    S = 20 * np.log10(np.abs(X).T + 1e-6)
    dur = len(x) / FS
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=110)
    ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
              extent=[0, dur, 0, FS / 2000])
    grey = "#8a8a90"  # readable on light and dark pages
    ax.set_xlabel("time (s)", color=grey)
    ax.set_ylabel("kHz", color=grey)
    ax.tick_params(colors=grey)
    for s in ax.spines.values():
        s.set_color(grey)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def process(raw, noise_kind, snr_db, seed):
    x = read_audio(raw)
    if len(x) < FS // 2:
        raise ValueError("recording is shorter than 0.5 s")
    rng = np.random.default_rng(seed)
    reference = None
    if noise_kind != "none":
        reference = x
        x = mix_at_snr(x, make_noise(noise_kind, len(x), rng), snr_db)

    ours = denoise(x, MODEL)
    base = spectral_subtraction(x)
    tracks = {"input": x, "ours": ours, "specsub": base}

    scale = 1 / max(1.0, max(np.max(np.abs(t)) for t in tracks.values()))
    ref_spec = 20 * np.log10(np.abs(stft(x)) + 1e-6)
    vmax = float(np.percentile(ref_spec, 99.5))
    vmin = vmax - 80

    out = {"seconds": len(x) / FS, "has_reference": reference is not None, "tracks": {}}
    for name, t in tracks.items():
        entry = {"wav": to_wav_b64(t, scale), "spec": spectrogram_b64(t, vmin, vmax)}
        entry.update(level_stats(t))
        if reference is not None:
            entry["snr_db"] = float(snr(reference, t))
            entry["stoi"] = float(stoi(reference, t, FS))
        out["tracks"][name] = entry
    if reference is not None:
        out["tracks"]["clean"] = {"wav": to_wav_b64(reference, scale),
                                  "spec": spectrogram_b64(reference, vmin, vmax)}
    return out


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/sample.wav" and os.path.exists(SAMPLE_FILE):
            with open(SAMPLE_FILE, "rb") as f:
                self._send(200, f.read(), "audio/wav")
        elif path == "/info":
            self._json(200, {"model": MODEL_FILE,
                             "dataset_noise": os.path.isdir(os.path.join(DATA, "noisy_trainset_28spk_wav")),
                             "sample": os.path.exists(SAMPLE_FILE)})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/denoise":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > MAX_BYTES:
            return self._json(400, {"error": "audio missing or larger than 60 MB"})
        raw = self.rfile.read(n)
        q = parse_qs(url.query)
        noise = q.get("noise", ["none"])[0]
        snr_db = float(q.get("snr", ["5"])[0])
        seed = int(q.get("seed", [random.randrange(1 << 30)])[0])
        try:
            self._json(200, process(raw, noise, snr_db, seed))
        except Exception as e:  # report any decoding/processing error to the page
            self._json(400, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s\n" % (fmt % args))


if __name__ == "__main__":
    print(f"open http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
