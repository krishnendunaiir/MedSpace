import torch
from torch.utils.data import TensorDataset

from src.model import make
from src.train import fit


def test_fit(tmp_path):
    ds = TensorDataset(torch.randn(8, 3, 224, 224), torch.tensor([0, 1] * 4))
    out = tmp_path / "b.pt"
    best = fit(make(False), ds, ds, torch.ones(2), epochs=1, bs=4, out=str(out))
    assert 0 <= best <= 1 and out.exists()
