from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def lines(f):
    return (ROOT / f).read_text(encoding="utf-8-sig").splitlines()


def test_packages():
    assert lines("packages.txt") == ["libgl1"]


def test_requirements_pinned():
    r = lines("requirements.txt")
    assert r[0].startswith("--extra-index-url") and "whl/cpu" in r[0]
    assert any(x.startswith("torch==") and x.endswith("+cpu") for x in r)
    assert any(x.startswith("torchvision==") and x.endswith("+cpu") for x in r)
    assert "opencv-python-headless" in r and "streamlit" in r
