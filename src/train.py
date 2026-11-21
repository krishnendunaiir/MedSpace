import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from src.data import build, loader
from src.model import class_weights, make
from src.utils import device, seed_all


def evaluate(m, dl, dev):
    m.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            ps += m(x.to(dev)).softmax(1)[:, 1].cpu().tolist()
            ys += y.tolist()
    return roc_auc_score(ys, ps)


def fit(m, tr, va, w, epochs=3, lr=1e-4, bs=32, out="best.pt"):
    dev = device()
    m.to(dev)
    lf = nn.CrossEntropyLoss(weight=w.to(dev))
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    tl, vl = loader(tr, bs, True), loader(va, bs)
    best = -1.0
    for e in range(epochs):
        m.train()
        for x, y in tl:
            opt.zero_grad()
            lf(m(x.to(dev)), y.to(dev)).backward()
            opt.step()
        auc = evaluate(m, vl, dev)
        print(f"epoch {e + 1} val_auroc {auc:.4f}")
        if auc > best:
            best = auc
            torch.save(m.state_dict(), out)
    return best


if __name__ == "__main__":
    seed_all()
    tr, va, _ = build("data/chest_xray/chest_xray")
    fit(make(), tr, va, class_weights(tr.items))
