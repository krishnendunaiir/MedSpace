import numpy as np
import torch
from PIL import Image

from src.cfg import TRIM, WEIGHTS
from src.data import tf
from src.gradcam import cam, overlay
from src.model import make


def load(path=WEIGHTS):
    m = make(False)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def prep(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return tf(False, TRIM)(img)[None]


def classify(m, x):
    with torch.no_grad():
        p = m(x).softmax(1)[0].tolist()
    return {"NORMAL": p[0], "PNEUMONIA": p[1]}


def explain(m, x):
    return overlay(x[0], cam(m, x)[0])


def run_stages(m, img, cb=None):
    note = cb or (lambda v, t: None)
    note(0.25, "Preprocessing image...")
    x = prep(img)
    note(0.5, "Classifying...")
    probs = classify(m, x)
    note(0.8, "Generating Grad-CAM heatmap...")
    heat = explain(m, x)
    note(1.0, "Done")
    return probs, heat


def predict(m, img):
    return run_stages(m, img)
