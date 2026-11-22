from pathlib import Path
from unittest.mock import patch
from PIL import Image
from streamlit.testing.v1 import AppTest
from src.medimg import load_image

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def test_bundled_examples_and_credit():
    assert "CC BY 4.0" in Path(APP).read_text(encoding="utf-8")
    files = sorted(Path("assets/examples").glob("*.jpeg"))
    assert len(files) == 6
    for f in files:
        img, _ = load_image(f.read_bytes(), f.name)
        assert img.size[0] > 0


def test_example_button_runs_inference():
    dummy_cam = Image.new("RGB", (64, 64))
    with patch("src.weights.ensure", return_value=Path("best.pt")), patch(
        "src.app.load", return_value=object()
    ), patch(
        "src.app.run_stages",
        return_value=({"NORMAL": 0.92, "PNEUMONIA": 0.08}, dummy_cam),
    ):
        at = AppTest.from_file(APP).run(timeout=120)
        assert len(at.button) >= 6
        at.button[0].click().run(timeout=120)
        assert any("Model output: NORMAL" in s.value for s in at.subheader)
