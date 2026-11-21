import numpy as np
from PIL import Image

from src.data import folder, loader


def make(root):
    for c in ("NORMAL", "PNEUMONIA"):
        (root / c).mkdir(parents=True)
        for i in range(2):
            a = np.random.randint(0, 255, (300, 400), dtype=np.uint8)
            Image.fromarray(a).save(root / c / f"{i}.png")


def test_folder(tmp_path):
    make(tmp_path)
    ds = folder(tmp_path, train=True)
    assert ds.classes == ["NORMAL", "PNEUMONIA"] and len(ds) == 4
    x, y = next(iter(loader(ds, 2)))
    assert x.shape == (2, 3, 224, 224) and y.shape == (2,)
