import numpy as np

from src.app import run_stages
from src.model import make


def test_stages():
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    seen = []
    p, o = run_stages(make(False).eval(), img, lambda v, t: seen.append((v, t)))
    vals = [v for v, _ in seen]
    assert vals == sorted(vals) and vals[-1] == 1.0 and len(seen) == 4
    assert set(p) == {"NORMAL", "PNEUMONIA"} and o.shape == (224, 224, 3)


def test_no_callback():
    p, _ = run_stages(make(False).eval(), np.zeros((64, 64, 3), dtype=np.uint8))
    assert abs(sum(p.values()) - 1) < 1e-5
