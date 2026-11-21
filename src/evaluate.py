import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.data import build, loader
from src.model import make
from src.utils import device


def predict(m, dl, dev):
    m.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            ps += m(x.to(dev)).softmax(1)[:, 1].cpu().tolist()
            ys += y.tolist()
    return ys, ps


def pick_threshold(y, p):
    f, t, th = roc_curve(y, p)
    return float(th[(t - f).argmax()])


def metrics(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, [int(v >= t) for v in p], labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(y, p),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


if __name__ == "__main__":
    dev = device()
    _, va, te = build("data/chest_xray/chest_xray")
    m = make(False)
    m.load_state_dict(torch.load("best.pt", map_location=dev))
    m.to(dev)
    t = pick_threshold(*predict(m, loader(va), dev))
    print("threshold", round(t, 4))
    print(metrics(*predict(m, loader(te), dev), t))
