import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402


def roc_png(y, p, path):
    f, t, _ = roc_curve(y, p)
    plt.figure()
    plt.plot(f, t)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(path)
    plt.close()


def calib_png(y, p, path, bins=10):
    a, b = calibration_curve(y, p, n_bins=bins)
    plt.figure()
    plt.plot(b, a, "o-")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("predicted")
    plt.ylabel("observed")
    plt.savefig(path)
    plt.close()


def worst_fn(items, p, t, k=8):
    fn = sorted((pi, path) for (path, y), pi in zip(items, p) if y == 1 and pi < t)
    return [path for _, path in fn[:k]]


if __name__ == "__main__":
    from src.data import build, loader
    from src.evaluate import pick_threshold, predict
    from src.model import make
    from src.utils import device

    dev = device()
    _, va, te = build("data/chest_xray/chest_xray")
    m = make(False)
    m.load_state_dict(torch.load("best.pt", map_location=dev))
    m.to(dev)
    t = pick_threshold(*predict(m, loader(va), dev))
    y, p = predict(m, loader(te), dev)
    os.makedirs("outputs/fn", exist_ok=True)
    roc_png(y, p, "outputs/roc.png")
    calib_png(y, p, "outputs/calib.png")
    for i, f in enumerate(worst_fn(te.items, p, t)):
        shutil.copy(f, f"outputs/fn/{i}_{os.path.basename(f)}")
