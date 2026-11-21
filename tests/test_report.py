import json

import pytest

from src.report import summarize


def test_summarize():
    r = summarize([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], [0.6, 0.2, 0.8, 0.4], [0.2, 0.6, 0.1, 0.3])
    assert r["threshold"] == pytest.approx(0.8)
    assert r["auroc"] == pytest.approx(0.75)
    assert r["sensitivity"] == 0.5 and r["specificity"] == 1.0
    assert r["cm"] == [[2, 0], [1, 1]]
    assert r["border_share"] == pytest.approx(0.3) and r["border_over_half"] == 1
    assert r["n_border"] == 4 and r["n_empty"] == 0
    json.dumps(r)


def test_nan_borders():
    nan = float("nan")
    r = summarize([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], [0.6, 0.2, 0.8, 0.4], [0.2, nan, 0.6])
    assert r["border_share"] == pytest.approx(0.4) and r["border_over_half"] == 1
    assert r["n_border"] == 2 and r["n_empty"] == 1
    assert "NaN" not in json.dumps(r)
