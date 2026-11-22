import tomllib
from pathlib import Path

import numpy as np
from PIL import Image

from src.logo import make

ROOT = Path(__file__).resolve().parent.parent


def check(p):
    im = Image.open(p)
    a = np.asarray(im)
    assert im.size == (512, 512) and im.mode == "RGBA"
    assert a[0, 0, 3] == 0 and a[256, 256, 3] == 255
    white = (a[..., :3].min(-1) > 235) & (a[..., 3] > 200)
    assert white.sum() > 5000


def test_make(tmp_path):
    p = tmp_path / "i.png"
    make(str(p))
    check(p)


def test_committed_assets():
    check(ROOT / "assets" / "icon.png")


def test_theme():
    t = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert t["theme"]["light"]["primaryColor"] == "#0E7490"
    assert t["theme"]["dark"]["primaryColor"] == "#22D3EE"
