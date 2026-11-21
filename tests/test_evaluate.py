from src.evaluate import metrics, pick_threshold


def test_perfect():
    y, p = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    t = pick_threshold(y, p)
    m = metrics(y, p, t)
    assert 0.2 < t <= 0.8
    assert m["auroc"] == 1 and m["sensitivity"] == 1 and m["specificity"] == 1


def test_mixed():
    m = metrics([0, 0, 1, 1], [0.6, 0.2, 0.8, 0.4], 0.5)
    assert m["auroc"] == 0.75
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert m["cm"] == [[1, 1], [1, 1]]
