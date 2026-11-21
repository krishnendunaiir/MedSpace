import numpy as np
import torch
from PIL import Image

from src.data import tf
from src.gradcam import cam, overlay
from src.model import make


def load(path="best.pt"):
    m = make(False)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def predict(m, img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    x = tf(False)(img)[None]
    with torch.no_grad():
        p = m(x).softmax(1)[0].tolist()
    return {"NORMAL": p[0], "PNEUMONIA": p[1]}, overlay(x[0], cam(m, x)[0])
