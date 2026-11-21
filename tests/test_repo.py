from pathlib import Path


def txt(p):
    return Path(p).read_bytes().decode("utf-8-sig", errors="ignore").replace("\x00", "")


def test_gitignore():
    t = txt(".gitignore").splitlines()
    for s in (".venv/", "data/", "*.pt", "space/", ".pytest_cache/", "__pycache__/"):
        assert s in t, s


def test_requirements():
    t = txt("requirements.txt").lower().splitlines()
    for s in ("torch", "torchvision", "numpy", "pillow", "scikit-learn", "grad-cam", "matplotlib", "gradio", "pytest"):
        assert s in t, s


def test_readme_state():
    t = txt("README.md")
    assert "TBD" not in t
    assert "is nan" not in t
    assert "Live demo" not in t
