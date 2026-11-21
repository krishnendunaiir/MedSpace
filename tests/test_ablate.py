import torch
from torch.utils.data import TensorDataset

from src.ablate import auroc_with, mask
from src.model import make


def test_mask():
    x = torch.ones(1, 3, 10, 10)
    b, c = mask(x, "border", 0.2), mask(x, "center", 0.2)
    assert b.sum() == 3 * 36 and c.sum() == 3 * 64 and x.sum() == 300
    assert b[0, 0, 0, 0] == 0 and b[0, 0, 5, 5] == 1
    assert c[0, 0, 0, 0] == 1 and c[0, 0, 5, 5] == 0


def test_auroc_with():
    ds = TensorDataset(torch.randn(8, 3, 224, 224), torch.tensor([0, 1] * 4))
    for mode in (None, "border", "center"):
        assert 0 <= auroc_with(make(False), ds, mode, "cpu") <= 1
