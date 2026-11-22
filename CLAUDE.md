# RNNoise DSP course project

## What this is
- DSP course project (EEE F434 / ECE F434, BITS Pilani, instructor Dr. Sandeep Joshi) by Rik Sengupta and Rohan Priyan Vassoudevane.
- We are re-implementing, in Python, J.-M. Valin, "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement" (RNNoise, arXiv:1709.08243, 2018).
- Deliverables: a working implementation, a demo video of at most 3 minutes (approach, results, learning), and a viva (tentatively last week of November 2026). A research paper is optional.
- We are graded mainly through the viva, so we must be able to explain every line in `src/`.

## How to work with us
- Teach, don't just do. Go one step at a time: give a small piece of code for us to add, tell us how to run it, say what output to expect, then explain what each line does mathematically and which part of the paper it implements.
- Do not write or rewrite whole files in `src/` unless we explicitly ask.
- Keep explanations simple, with an everyday analogy, and end each step with one short check question.
- When we hit an error, explain what caused it before fixing it.
- Environment: Windows, VS Code, Command Prompt terminal. Run Python as `venv\Scripts\python`.

## Layout
- `src/` — our implementation, one file per stage (plan below)
- `run_rnnoise.py` + `rnnoise_demo.exe` — the official RNNoise v0.1 (built from github.com/xiph/rnnoise, tag v0.1), used only as the reference/benchmark. Usage: `venv\Scripts\python run_rnnoise.py <noisy.wav> <out.wav>`
- `data/` — Valentini-Botinhao noisy speech database (Edinburgh DataShare, 48 kHz mono): `clean_testset_wav/` and `noisy_testset_wav/`, 824 parallel files each, named like `p232_001.wav`. Not in git.
- `outputs/` — generated audio. Not in git.
- `explore.ipynb` — scratch notebook for experiments and listening
- `setup.bat`, `vscode_config/` — one-click Windows environment setup

## Key numbers (from the paper)
- 48 kHz; frame 960 samples (20 ms); hop 480 (10 ms); Vorbis window, eq. (1): w(n) = sin[(π/2)·sin²(πn/N)], satisfies Princen–Bradley
- 481 FFT bins, 50 Hz apart; 22 triangular Bark-like bands (band edges = `eband5ms` in v0.1 `src/denoise.c`)
- 42 input features; network Dense 24 → GRU 24 (VAD) → GRU 48 → GRU 96 → Dense 22 gains (+ VAD output); 87,503 weights in the paper (88,007 in PyTorch)
- Loss (g^γ − ĝ^γ)² with γ = 1/2; gain smoothing g̃ = max(0.6·g̃_prev, ĝ)

## Plan for src/
| File | Purpose | When |
| --- | --- | --- |
| stft.py | framing, Vorbis window, FFT, overlap-add | Week 1 |
| bands.py | 22 bands, band energies, gain interpolation | Week 1 |
| ideal_gains.py | ideal gains from clean + noisy (eq. 3) | Week 1 |
| baseline.py | spectral subtraction baseline | Week 1 |
| pitch.py | pitch period, pitch correlation, comb filter (eqs. 5–6) | Week 2 |
| features.py | 42 features | Week 2 |
| dataset.py | training data mixing and augmentation (eq. 7) | Week 2 |
| model.py | the GRU network (Fig. 3) | Week 2 |
| train.py | training with the custom loss | Week 3 |
| denoise.py | full pipeline: noisy wav in, clean wav out | Week 3 |
| evaluate.py | PESQ/STOI, comparison tables, spectrograms | Week 3 |

Proposed split: Rik owns the DSP files (stft, bands, ideal_gains, baseline, pitch, features); Rohan owns the model files (dataset, model, train, evaluate). `denoise.py` is joint.

## Status
- Setup done; official RNNoise verified on real speech; test set downloaded.
- 2026-11-05: `src/stft.py` done — `vorbis_window`, `stft` (pad HOP zeros before / FRAME after, frame, window, rfft → frames × 481 bins) and `istft` (irfft, window, overlap-add, trim). Round-trip error 4.4e-16 on p232_001.
- 2026-11-21: done — `bands.py` (22 triangular bands, band_energy eq. 2, interp_gains eq. 4; ideal-gain demo), `baseline.py` (spectral subtraction + SNR/STOI scoreboard), `features.py` (34 features: 22 BFCC + 6 deltas + 6 delta-deltas; pitch features skipped on purpose; targets = ideal gains + VAD), `dataset.py`, `model.py` (Fig. 3 GRU net, 84,359 weights), `train.py` (loss (√g − √ĝ)² + 0.5·BCE, 10% held out), `denoise.py` (includes gain smoothing eq. 9, λ = 0.6). Practice run on 100 test files: loss 0.45 → 0.28 in 30 epochs.
- 2026-11-22: full training done on a friend's RTX 5080 (28-speaker set, 3060 train / 339 check sequences, 100 epochs): loss 0.35 → 0.152 train, checking 0.283 → 0.178 (best epoch 98). `rnnoise_model.pt` + `train_log.txt` in repo; `demo/` local web page added by the friend (calls src/ only).
- 2026-11-22: post-processing added in `denoise.py` (not in the paper): gain floor 0.1 (≈27% less gain flicker, STOI 0.890 → 0.896) and speech loudness normalisation to −20 dBFS using the model's VAD, capped at 0.99 peak.
- Results on 21 unseen test files (before post-processing): SNR / STOI / pause-noise — noisy 6.6 / 0.873 / 0; spectral subtraction 8.5 / 0.870 / −2.6 dB; official RNNoise 12.0 / 0.885 / −14.1 dB; ours 12.6 / 0.890 / −11.9 dB; ideal gains 13.7 / 0.959 / −13.5 dB. Describe as comparable to official RNNoise, not better.
- Next: final evaluation on more test files with final settings (STOI + SI-SNR, since loudness normalisation breaks plain SNR), spectrogram figures, loss graph, video, viva prep.
- Update this section as steps are finished.

## Notes
- `pesq` has no Windows wheel; compute PESQ later on Google Colab. `pystoi` works locally.
- Never commit `venv/`, `data/` or `outputs/`.
