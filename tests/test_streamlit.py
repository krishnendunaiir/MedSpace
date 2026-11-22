from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.weights import ensure

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def test_existing(tmp_path):
    f = tmp_path / "w.pt"
    f.write_bytes(b"x")
    assert ensure(str(f)) == str(f)


def test_missing_no_url(tmp_path):
    with pytest.raises(FileNotFoundError):
        ensure(str(tmp_path / "w.pt"))


def test_download(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"abc")
    dst = tmp_path / "dst.pt"
    assert ensure(str(dst), src.as_uri()) == str(dst)
    assert dst.read_bytes() == b"abc" and not (tmp_path / "dst.pt.part").exists()


def test_app_runs():
    at = AppTest.from_file(APP).run(timeout=120)
    assert not at.exception
    assert "Chest X-ray classifier" in at.markdown[0].value
    assert at.radio[0].options == ["Upload / drag & drop", "Webcam", "Paste from clipboard"]
    assert any("light and dark" in c.value for c in at.caption)
