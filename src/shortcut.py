import random

import torch


def border_frac(g, m=0.1):
    k = int(g.shape[-1] * m)
    inner = g[..., k:-k, k:-k].sum((-2, -1))
    return 1 - inner / g.sum((-2, -1))


if __name__ == "__main__":
    from src.data import build
    from src.gradcam import cam
    from src.model import make
    from src.utils import device

    dev = device()
    te = build("data/chest_xray/chest_xray")[2]
    m = make(False)
    m.load_state_dict(torch.load("best.pt", map_location=dev))
    m.to(dev)
    ids = random.Random(0).sample(range(len(te)), 200)
    r = []
    for i in range(0, len(ids), 32):
        x = torch.stack([te[j][0] for j in ids[i : i + 32]]).to(dev)
        r += border_frac(cam(m, x)).tolist()
    print(f"mean border share {sum(r) / len(r):.2f} (uniform would be 0.36)")
    print(f"images with border share > 0.5: {sum(v > 0.5 for v in r)} of {len(r)}")
