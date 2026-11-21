import torch
import torch.nn as nn

N_FEATURES = 34
N_BANDS = 22


class RNNoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(N_FEATURES))
        self.register_buffer("std", torch.ones(N_FEATURES))
        self.dense_in = nn.Linear(N_FEATURES, 24)
        self.vad_gru = nn.GRU(24, 24, batch_first=True)
        self.vad_out = nn.Linear(24, 1)
        self.noise_gru = nn.GRU(24 + 24 + N_FEATURES, 48, batch_first=True)
        self.denoise_gru = nn.GRU(24 + 48 + N_FEATURES, 96, batch_first=True)
        self.gains_out = nn.Linear(96, N_BANDS)

    def forward(self, x):
        x = (x - self.mean) / self.std
        d = torch.tanh(self.dense_in(x))
        v, _ = self.vad_gru(d)
        vad = torch.sigmoid(self.vad_out(v)).squeeze(-1)
        n, _ = self.noise_gru(torch.cat([d, v, x], dim=-1))
        g, _ = self.denoise_gru(torch.cat([v, n, x], dim=-1))
        gains = torch.sigmoid(self.gains_out(g))
        return gains, vad


if __name__ == "__main__":
    model = RNNoise()
    x = torch.randn(2, 100, N_FEATURES)
    gains, vad = model(x)
    print("input (sequences, frames, features):", tuple(x.shape))
    print("gains out:", tuple(gains.shape), " vad out:", tuple(vad.shape))
    print("trainable weights:", sum(p.numel() for p in model.parameters()))