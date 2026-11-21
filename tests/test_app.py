import numpy as np
import torch

from src.app import load, predict
from src.model import make


def test_predict():
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    p, o = predict(make(False).eval(), img)
    assert set(p) == {"NORMAL", "PNEUMONIA"} and abs(sum(p.values()) - 1) < 1e-5
    assert o.shape == (224, 224, 3) and o.dtype == np.uint8


def test_load(tmp_path):
    f = tmp_path / "w.pt"
    torch.save(make(False).state_dict(), f)
    assert not load(str(f)).training
