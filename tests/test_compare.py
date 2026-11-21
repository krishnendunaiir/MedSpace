import torch
from torch.utils.data import TensorDataset

from src.compare import table
from src.model import make


def test_table():
    ds = TensorDataset(torch.randn(8, 3, 224, 224), torch.tensor([0, 1] * 4))
    r = table(make(False), ds, "cpu")
    assert "normal" in r and len(r) == 6 and all(0 <= v <= 1 for v in r.values())
