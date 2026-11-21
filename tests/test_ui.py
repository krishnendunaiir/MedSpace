import gradio as gr
import numpy as np

from src.model import make
from src.ui import build_ui


def test_ui():
    ui = build_ui(make(False).eval())
    assert isinstance(ui, gr.Interface)
    p, o = ui.fn(np.zeros((64, 64, 3), dtype=np.uint8))
    assert set(p) == {"NORMAL", "PNEUMONIA"} and o.shape == (224, 224, 3)


def test_none():
    assert build_ui(make(False).eval()).fn(None) == (None, None)
