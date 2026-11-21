import sys
import numpy as np
import torch
from model import RNNoise

SEQ = 1000


def to_sequences(a):
    n = len(a) // SEQ
    return torch.tensor(a[: n * SEQ].reshape(n, SEQ, *a.shape[1:]))


def gain_loss(pred, true):
    return ((torch.sqrt(true) - torch.sqrt(pred + 1e-8)) ** 2).mean()


if __name__ == "__main__":
    data_file, model_file = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("training on:", device)

    d = np.load(data_file)
    F, G, V = to_sequences(d["features"]), to_sequences(d["gains"]), to_sequences(d["vad"])
    n_val = max(1, len(F) // 10)
    train_idx, val_idx = torch.arange(n_val, len(F)), torch.arange(n_val)
    print(f"sequences of {SEQ} frames: {len(train_idx)} for training, {n_val} for checking")

    model = RNNoise().to(device)
    feats = d["features"]
    model.mean.copy_(torch.tensor(feats.mean(axis=0)))
    model.std.copy_(torch.tensor(feats.std(axis=0) + 1e-5))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCELoss()
    batch = 32

    for epoch in range(1, epochs + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            x, g, v = F[idx].to(device), G[idx].to(device), V[idx].to(device)
            pg, pv = model(x)
            loss = gain_loss(pg, g) + 0.5 * bce(pv, v)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            x, g, v = F[val_idx].to(device), G[val_idx].to(device), V[val_idx].to(device)
            pg, pv = model(x)
            val = (gain_loss(pg, g) + 0.5 * bce(pv, v)).item()
        print(f"epoch {epoch:3d}   training loss {total / len(train_idx):.4f}   checking loss {val:.4f}")

    torch.save(model.state_dict(), model_file)
    print("saved", model_file)