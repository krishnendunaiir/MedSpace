import torch
from sklearn.metrics import roc_auc_score

from src.data import loader


def mask(x, mode, frac=0.15):
    k = max(1, int(x.shape[-1] * frac))
    y = x.clone()
    if mode == "border":
        y[..., :k, :] = 0
        y[..., -k:, :] = 0
        y[..., :, :k] = 0
        y[..., :, -k:] = 0
    else:
        y[..., k:-k, k:-k] = 0
    return y


def auroc_with(m, ds, mode, dev, frac=0.15):
    m.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader(ds, 64):
            if mode:
                x = mask(x, mode, frac)
            ps += m(x.to(dev)).softmax(1)[:, 1].cpu().tolist()
            ys += y.tolist()
    return roc_auc_score(ys, ps)


if __name__ == "__main__":
    from src.data import build
    from src.model import make
    from src.utils import device

    dev = device()
    te = build("data/chest_xray/chest_xray")[2]
    m = make(False)
    m.load_state_dict(torch.load("best.pt", map_location=dev))
    m.to(dev)
    for mode in (None, "border", "center"):
        print(mode, round(auroc_with(m, te, mode, dev), 4))
