import shutil
from pathlib import Path

import gradio as gr

FILES = ["app.py", *(f"src/{n}.py" for n in ("__init__", "cfg", "data", "split", "model", "gradcam", "app", "ui"))]
REQ = "torch\ntorchvision\nnumpy\npillow\nscikit-learn\ngrad-cam\ngradio\n"
README = """---
title: Chest X-ray Pneumonia Classifier
emoji: 🫁
sdk: gradio
sdk_version: {v}
app_file: app.py
pinned: false
---

Research demo: ResNet-18 pneumonia vs. normal classifier with Grad-CAM explanations.
Not a medical device and not for diagnosis.
"""


def make_space(out="space", weights="best_trim.pt"):
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    for f in FILES:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, out / f)
    shutil.copy(weights, out / Path(weights).name)
    (out / "requirements.txt").write_text(REQ, encoding="utf-8")
    (out / "README.md").write_text(README.format(v=gr.__version__), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(make_space())
