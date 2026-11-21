import numpy as np
import torch

from src.gradcam import cam, overlay, save
from src.model import make


def test_cam():
    x = torch.randn(2, 3, 224, 224)
    g = cam(make(False), x)
    assert g.shape == (2, 224, 224) and g.min() >= 0 and g.max() <= 1 + 1e-6


def test_overlay_save(tmp_path):
    x = torch.randn(3, 224, 224)
    g = cam(make(False), x[None])[0]
    o = overlay(x, g)
    assert o.shape == (224, 224, 3) and o.dtype == np.uint8
    p = tmp_path / "o.png"
    save(x, g, str(p))
    assert p.stat().st_size > 0
