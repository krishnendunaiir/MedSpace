import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data import MEAN, STD


def cam(m, x, cls=1):
    m.eval()
    with GradCAM(model=m, target_layers=[m.layer4[-1]]) as c:
        return c(input_tensor=x, targets=[ClassifierOutputTarget(cls)] * len(x))


def overlay(x, g):
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    img = (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy().astype("float32")
    return show_cam_on_image(img, g, use_rgb=True)


def save(x, g, path):
    Image.fromarray(overlay(x, g)).save(path)


if __name__ == "__main__":
    import os

    from src.data import build
    from src.model import make
    from src.utils import device

    dev = device()
    te = build("data/chest_xray/chest_xray")[2]
    m = make(False)
    m.load_state_dict(torch.load("best.pt", map_location=dev))
    m.to(dev)
    os.makedirs("outputs/cam", exist_ok=True)
    ids = [i for c in (0, 1) for i in [j for j, (_, y) in enumerate(te.items) if y == c][:4]]
    x = torch.stack([te[i][0] for i in ids]).to(dev)
    for i, xi, gi in zip(ids, x, cam(m, x)):
        save(xi, gi, f"outputs/cam/{te.items[i][1]}_{i}.png")
