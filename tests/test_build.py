from PIL import Image

from src.data import build, gather
from src.split import pid


def make(root):
    for i in range(30):
        s = ("train", "val", "test")[i % 3]
        for c, f in (("NORMAL", "IM-{:04d}-{:04d}.png"), ("PNEUMONIA", "person{}_bacteria_{}.png")):
            d = root / s / c
            d.mkdir(parents=True, exist_ok=True)
            for j in range(2):
                Image.new("L", (32, 32)).save(d / f.format(i, j))


def groups(ds):
    return {pid(p) for p, _ in ds.items}


def test_build(tmp_path):
    make(tmp_path)
    assert len(gather(str(tmp_path))) == 120
    tr, va, te = build(str(tmp_path))
    assert len(tr) + len(va) + len(te) == 120
    assert not (groups(tr) & groups(va)) and not (groups(tr) & groups(te)) and not (groups(va) & groups(te))
    assert {y for _, y in tr.items} == {0, 1}
    x, y = tr[0]
    assert x.shape == (3, 224, 224)
