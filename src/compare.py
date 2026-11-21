from src.ablate import auroc_with
from src.sweep import CASES


def table(m, ds, dev):
    out = {"normal": auroc_with(m, ds, None, dev)}
    out.update({n: auroc_with(m, ds, mode, dev, f) for n, mode, f in CASES})
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path

    import torch

    from src.data import build
    from src.model import make
    from src.utils import device

    dev = device()
    root = "data/chest_xray/chest_xray"
    res = {}
    for name, w, trim in (("baseline", "best.pt", 0.0), ("trimmed", "best_trim.pt", 0.10)):
        te = build(root, trim=trim)[2]
        m = make(False)
        m.load_state_dict(torch.load(w, map_location=dev))
        m.to(dev)
        res[name] = table(m, te, dev)
    Path("outputs/ablation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"{'case':34s} {'baseline':>9s} {'trimmed':>9s}")
    for k in res["baseline"]:
        print(f"{k:34s} {res['baseline'][k]:9.4f} {res['trimmed'][k]:9.4f}")
