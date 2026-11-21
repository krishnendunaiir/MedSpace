from PIL import Image

from src.data import Trim, tf


def test_trim():
    im = Image.new("L", (100, 200))
    assert Trim(0.1)(im).size == (80, 160)
    assert Trim(0.0)(im).size == (100, 200)


def test_tf_trim():
    im = Image.new("L", (100, 200))
    assert tf(False, 0.1)(im).shape == (3, 224, 224)
    assert tf(False)(im).shape == (3, 224, 224)
