import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.app import label, run_stages
from src.calibrate import calp, ece, fit, fit_platt, fit_temperature, load_cal, logit_diff, nll, pick_d_threshold, report, sigmoid
from src.data import loader
from src.model import make


def sim(n=6000, k=5.0, seed=0):
    r = np.random.default_rng(seed)
    z = r.normal(0, 2, n)
    y = (r.random(n) < sigmoid(z)).astype(int)
    return k * z, y


def test_sigmoid():
    assert sigmoid(0) == 0.5 and sigmoid(1000) <= 1 and sigmoid(-1000) >= 0


def test_temperature_recovers_scale():
    d, y = sim()
    t = fit_temperature(d, y)
    assert 3.5 < t < 7
    assert nll(d, y, t) < nll(d, y)


def test_platt_fixes_bias():
    r = np.random.default_rng(1)
    z = r.normal(0, 2, 8000)
    y = (r.random(8000) < sigmoid(z)).astype(int)
    d = 3 * z - 2
    a, b = fit_platt(d, y)
    assert abs(a - 1 / 3) < 0.05 and abs(b - 2 / 3) < 0.25
    t = fit_temperature(d, y)
    assert ece(sigmoid(a * d + b), y) < ece(sigmoid(d / t), y)


def test_ece():
    assert abs(ece([0.9] * 10, [1] * 5 + [0] * 5) - 0.4) < 1e-9
    assert ece([0.0, 1.0], [0, 1]) == 0


def test_threshold_and_fit():
    assert -2 < pick_d_threshold([-3, -2, 2, 3], [0, 0, 1, 1]) <= 2
    d, y = sim()
    c = fit(d, y)
    assert abs(c["threshold"] - sigmoid(c["a"] * c["d_threshold"] + c["b"])) < 1e-12
    r = report(c, d, y)
    assert r["ece_after"] < r["ece_before"] and r["nll_after"] < r["nll_before"]
    assert 0 <= r["sensitivity"] <= 1 and sum(map(sum, r["cm"])) == len(y)


def test_calp():
    assert abs(calp({"a": 0.5, "b": 1.0}, 0) - sigmoid(1.0)) < 1e-12
    assert abs(calp({"temperature": 2.0}, 4) - sigmoid(2.0)) < 1e-12


def test_logit_diff():
    ds = TensorDataset(torch.randn(8, 3, 224, 224), torch.tensor([0, 1] * 4))
    y, d = logit_diff(make(False), loader(ds, 4), "cpu")
    assert len(y) == len(d) == 8 and np.isfinite(d).all()


def test_app_with_cal():
    m = make(False).eval()
    img = np.random.randint(0, 255, (80, 90, 3), dtype=np.uint8)
    p0, _ = run_stages(m, img)
    for cal in ({"temperature": 5.0, "threshold": 0.3}, {"a": 0.5, "b": 1.0, "threshold": 0.3}):
        p1, _ = run_stages(m, img, cal=cal)
        assert abs(sum(p1.values()) - 1) < 1e-6
    p2, _ = run_stages(m, img, cal={"temperature": 5.0, "threshold": 0.3})
    assert abs(p2["PNEUMONIA"] - 0.5) <= abs(p0["PNEUMONIA"] - 0.5) + 1e-9


def test_label():
    p = {"NORMAL": 0.7, "PNEUMONIA": 0.3}
    assert label(p) == "NORMAL" and label(p, {"threshold": 0.3}) == "PNEUMONIA"
    assert label(p, {"threshold": 0.31}) == "NORMAL"


def test_load_cal(tmp_path):
    assert load_cal(str(tmp_path / "x.json")) is None
    f = tmp_path / "c.json"
    f.write_text('{"temperature": 2.0, "threshold": 0.4}', encoding="utf-8")
    assert load_cal(str(f))["temperature"] == 2.0
