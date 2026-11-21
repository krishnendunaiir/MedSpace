import torch
from torch import nn

from src.model import class_weights, make


def test_make():
    assert make(False)(torch.zeros(2, 3, 224, 224)).shape == (2, 2)


def test_weights():
    w = class_weights([("a", 0)] + [("b", 1)] * 3)
    assert torch.allclose(w, torch.tensor([2.0, 4 / 6]))


def test_loss_step():
    m = make(False)
    w = class_weights([("a", 0)] + [("b", 1)] * 3)
    loss = nn.CrossEntropyLoss(weight=w)(m(torch.zeros(2, 3, 224, 224)), torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)
