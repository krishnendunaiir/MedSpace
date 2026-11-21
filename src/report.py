from src.cfg import TRIM, WEIGHTS
import json
import os

from src.evaluate import metrics, pick_threshold


def summarize(yv, pv, yt, pt, borders):
    t = pick_threshold(yv, pv)
    m = metrics(yt, pt, t)
    b = [v for v in borders if v == v]
    return {
        "threshold": float(t),
        "auroc": float(m["auroc"]),
        "sensitivity": float(m["sensitivity"]),
        "specificity": float(m["specificity"]),
        "cm": m["cm"],
        "border_share": sum(b) / len(b),
        "border_over_half": sum(v > 0.5 for v in b),
        "n_border": len(b),
        "n_empty": len(borders) - len(b),
    }


if __name__ == "__main__":
    import random

    import torch

    from src.data import build, loader
    from src.evaluate import predict
    from src.gradcam import cam
    from src.model import make
    from src.shortcut import border_frac
    from src.utils import device

    dev = device()
    _, va, te = build("data/chest_xray/chest_xray", trim=TRIM)
    m = make(False)
    m.load_state_dict(torch.load(WEIGHTS, map_location=dev))
    m.to(dev)
    yv, pv = predict(m, loader(va), dev)
    yt, pt = predict(m, loader(te), dev)
    ids = random.Random(0).sample(range(len(te)), 200)
    b = []
    for i in range(0, len(ids), 32):
        x = torch.stack([te[j][0] for j in ids[i : i + 32]]).to(dev)
        b += border_frac(cam(m, x)).tolist()
    os.makedirs("outputs", exist_ok=True)
    r = summarize(yv, pv, yt, pt, b)
    with open("outputs/results.json", "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2)
    print(json.dumps(r, indent=2))
