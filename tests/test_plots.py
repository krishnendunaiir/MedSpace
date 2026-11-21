from src.plots import calib_png, roc_png, worst_fn


def test_pngs(tmp_path):
    y, p = [0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9]
    r, c = tmp_path / "r.png", tmp_path / "c.png"
    roc_png(y, p, str(r))
    calib_png(y, p, str(c), 2)
    assert r.stat().st_size > 0 and c.stat().st_size > 0


def test_worst_fn():
    items = [("a", 1), ("b", 1), ("c", 0), ("d", 1)]
    assert worst_fn(items, [0.1, 0.3, 0.2, 0.9], 0.5, 2) == ["a", "b"]
