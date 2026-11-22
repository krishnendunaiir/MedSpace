import numpy as np
import torch
from PIL import Image

from src.calibrate import calp, sigmoid
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


def classify(m, x, cal=None):
    with torch.no_grad():
        z = m(x)[0]
    d = (z[1] - z[0]).item()
    p1 = float(calp(cal, d) if cal else sigmoid(d))
    return {"NORMAL": 1 - p1, "PNEUMONIA": p1}


def label(probs, cal=None):
    t = cal["threshold"] if cal else 0.5
    return "PNEUMONIA" if probs["PNEUMONIA"] >= t else "NORMAL"


def explain(m, x):
    return overlay(x[0], cam(m, x)[0])


def run_stages(m, img, cb=None, cal=None):
    note = cb or (lambda v, t: None)
    note(0.25, "Preprocessing image...")
    x = prep(img)
    note(0.5, "Classifying...")
    probs = classify(m, x, cal)
    note(0.8, "Generating Grad-CAM heatmap...")
    heat = explain(m, x)
    note(1.0, "Done")
    return probs, heat


def predict(m, img):
    return run_stages(m, img)
