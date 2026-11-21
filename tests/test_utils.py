import torch

from src.utils import device, seed_all


def test_seed():
    seed_all(1)
    a = torch.rand(3)
    seed_all(1)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_device():
    assert device() in {"cuda", "mps", "cpu"}
