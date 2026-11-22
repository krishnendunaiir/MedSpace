import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

CAL_PATH = Path(__file__).resolve().parent.parent / "outputs" / "calibration.json"


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def nll_p(p, y):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def nll(d, y, t=1.0):
    return nll_p(sigmoid(np.asarray(d, float) / t), y)


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def fit_temperature(d, y):
    r = minimize_scalar(lambda s: nll(d, y, float(np.exp(s))), bounds=(-3, 4), method="bounded")
    return float(np.exp(r.x))


def fit_platt(d, y):
    d, y = np.asarray(d, float), np.asarray(y, float)
    r = minimize(
        lambda w: nll_p(sigmoid(w[0] * d + w[1]), y),
        x0=[1.0, 0.0],
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000},
    )
    return float(r.x[0]), float(r.x[1])


def calp(cal, d):
    a = cal["a"] if "a" in cal else 1.0 / cal["temperature"]
    return sigmoid(a * np.asarray(d, float) + cal.get("b", 0.0))


def pick_d_threshold(d, y):
    f, t, th = roc_curve(y, d)
    return float(th[(t - f).argmax()])


def fit(d, y):
    a, b = fit_platt(d, y)
    th = pick_d_threshold(d, y)
    return {"method": "platt", "a": a, "b": b, "d_threshold": th, "threshold": float(sigmoid(a * th + b))}


def load_cal(path=None):
    p = Path(path) if path else CAL_PATH
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8-sig"))


def logit_diff(m, dl, dev):
    m.eval()
    ys, ds = [], []
    with torch.no_grad():
        for x, y in dl:
            z = m(x.to(dev))
            ds += (z[:, 1] - z[:, 0]).cpu().tolist()
            ys += y.tolist()
    return ys, ds


def report(cal, d, y):
    d, y = np.asarray(d, float), np.asarray(y)
    pred = (d >= cal["d_threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p = calp(cal, d)
    return {
        "ece_before": ece(sigmoid(d), y),
        "ece_after": ece(p, y),
        "nll_before": nll(d, y),
        "nll_after": nll_p(p, y),
        "auroc": float(roc_auc_score(y, d)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


if __name__ == "__main__":
    from src.cfg import TRIM, WEIGHTS
    from src.data import build, loader
    from src.model import make
    from src.utils import device

    dev = device()
    _, va, te = build("data/chest_xray/chest_xray", trim=TRIM)
    m = make(False)
    m.load_state_dict(torch.load(WEIGHTS, map_location=dev))
    m.to(dev)
    yv, dv = logit_diff(m, loader(va), dev)
    yt, dt = logit_diff(m, loader(te), dev)
    cal = fit(dv, yv)
    cal["val_n"] = len(yv)
    cal["test"] = report(cal, dt, yt)
    t = fit_temperature(dv, yv)
    dta = np.asarray(dt)
    cal["test_temperature_only"] = {"temperature": t, "ece": ece(sigmoid(dta / t), yt), "nll": nll(dta, yt, t)}
    CAL_PATH.parent.mkdir(exist_ok=True)
    CAL_PATH.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(json.dumps(cal, indent=2))
