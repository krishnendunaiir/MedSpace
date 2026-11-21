from src.ablate import auroc_with

CASES = [
    ("frame 3%", "center", 0.03),
    ("frame 6%", "center", 0.06),
    ("frame 10%", "center", 0.10),
    ("frame 15%", "center", 0.15),
    ("center only (outer 25% blanked)", "border", 0.25),
]


def run(m, ds, dev, cases=CASES):
    return {n: auroc_with(m, ds, mode, dev, f) for n, mode, f in cases}


if __name__ == "__main__":
    import torch

    from src.data import build
    from src.model import make
    from src.utils import device

    dev = device()
    te = build("data/chest_xray/chest_xray")[2]
    m = make(False)
    m.load_state_dict(torch.load("best.pt", map_location=dev))
    m.to(dev)
    for n, v in run(m, te, dev).items():
        print(f"{n}: {v:.4f}")
