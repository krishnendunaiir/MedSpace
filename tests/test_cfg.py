import importlib
from pathlib import Path

import pytest

import src.cfg as c
from src.fill_readme import fill


def test_defaults(monkeypatch):
    monkeypatch.delenv("CXR_WEIGHTS", raising=False)
    monkeypatch.delenv("CXR_TRIM", raising=False)
    importlib.reload(c)
    assert c.WEIGHTS == "best_trim.pt" and c.TRIM == 0.10


def test_override(monkeypatch):
    monkeypatch.setenv("CXR_WEIGHTS", "best.pt")
    monkeypatch.setenv("CXR_TRIM", "0")
    importlib.reload(c)
    assert c.WEIGHTS == "best.pt" and c.TRIM == 0.0
    monkeypatch.undo()
    importlib.reload(c)


@pytest.mark.parametrize("f", ["report", "plots", "gradcam", "shortcut", "evaluate"])
def test_patched(f):
    t = Path(f"src/{f}.py").read_text(encoding="utf-8-sig")
    assert "from src.cfg import TRIM, WEIGHTS" in t
    assert "trim=TRIM" in t and "torch.load(WEIGHTS" in t
    assert 'torch.load("best.pt"' not in t


def test_app_patched():
    t = Path("src/app.py").read_text(encoding="utf-8-sig")
    assert "tf(False, TRIM)" in t and "path=WEIGHTS" in t


def test_refresh_numbers():
    r = {"auroc": 0.99, "sensitivity": 0.9, "specificity": 0.8, "cm": [[1, 2], [3, 4]], "threshold": 0.5, "border_share": 0.3, "border_over_half": 7, "n_border": 190}
    t = "| AUROC | 0.500 |\n| Confusion matrix [[TN, FP], [FN, TP]] | [[9, 9], [9, 9]] |\n(threshold 0.1)\nis 0.43 (56 of 194 sampled test images above 0.5; a uniform map would give 0.36)."
    o = fill(t, r)
    assert "| AUROC | 0.990 |" in o and "[[1, 2], [3, 4]]" in o and "(threshold 0.500)" in o
    assert "is 0.30 (7 of 190 sampled" in o and fill(o, r) == o
