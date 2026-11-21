import numpy as np

from src.shortcut import border_frac


def test_uniform():
    assert np.allclose(border_frac(np.ones((1, 10, 10))), 0.36)


def test_center_and_corner():
    g = np.zeros((2, 10, 10))
    g[0, 4:6, 4:6] = 1
    g[1, 0, 0] = 1
    assert np.allclose(border_frac(g), [0, 1])
