import json
import os

from src.evaluate import metrics, pick_threshold


def summarize(yv, pv, yt, pt, borders):
    t = pick_threshold(yv, pv)
    m = metrics(yt, pt, t)
    return {
        "threshold": float(t),
        "auroc": float(m["auroc"]),
        "sensitivity": float(m["sensitivity"]),
        "specificity": float(m["specificity"]),
        "cm": m["cm"],
        "border_share": sum(borders) / len(borders),
        "border_over_half": sum(b > 0.5 for b in borders),
        "n_border": len(borders),
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
    _, va, te = build("data/chest_xray/chest_xray")
    m = make(False)
    m.load_state_dict(torch.load("best.pt", map_location=dev))
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
