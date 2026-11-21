import torch
from torch.utils.data import TensorDataset

from src.model import make
from src.sweep import run


def test_run():
    ds = TensorDataset(torch.randn(8, 3, 224, 224), torch.tensor([0, 1] * 4))
    r = run(make(False), ds, "cpu", [("a", "center", 0.05), ("b", "border", 0.25)])
    assert set(r) == {"a", "b"} and all(0 <= v <= 1 for v in r.values())
