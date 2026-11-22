import os

import numpy as np
from PIL import Image, ImageDraw

S = 1024
OUT = 512
WHITE = (255, 255, 255, 255)


def background():
    top, bot = (14, 116, 144), (11, 37, 69)
    t = np.linspace(0, 1, S)[:, None]
    g = np.stack([np.repeat((top[i] + (bot[i] - top[i]) * t).astype(np.uint8), S, axis=1) for i in range(3)], axis=-1)
    im = Image.fromarray(g, "RGB").convert("RGBA")
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, S - 1, S - 1), radius=int(S * 0.22), fill=255)
    im.putalpha(m)
    return im


def lung(cx, cy, w, h, side):
    pts = []
    for k in range(180):
        t = 2 * np.pi * k / 180
        x = w * np.sin(t) * (0.55 + 0.45 * (1 - np.cos(t)) / 2)
        y = -h * np.cos(t)
        if x * side < 0:
            x *= 0.7
        pts.append((cx + x, cy + y))
    return pts


def glow(mask, cx, cy, sigma):
    yy, xx = np.mgrid[0:S, 0:S]
    t = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2)))
    orange, yellow = np.array([255, 122, 24]), np.array([255, 209, 102])
    rgb = orange[None, None, :] + (yellow - orange)[None, None, :] * t[..., None]
    a = (t * 235 * (np.asarray(mask) / 255.0)).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb.astype(np.uint8), a]), "RGBA")


def make(path, size=OUT):
    img = background()
    cy, w, h = 0.56 * S, 0.17 * S, 0.30 * S
    left = lung(0.30 * S, cy, w, h, -1)
    right = lung(0.70 * S, cy, w, h, 1)
    lm = Image.new("L", (S, S), 0)
    ImageDraw.Draw(lm).polygon(left, fill=255)
    lungs = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(lungs)
    d.polygon(left, fill=WHITE)
    d.polygon(right, fill=WHITE)
    img = Image.alpha_composite(img, lungs)
    img = Image.alpha_composite(img, glow(lm, 0.28 * S, 0.62 * S, 0.11 * S))
    tube = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(tube)
    wd = int(0.03 * S)
    for a, b in (((0.5, 0.17), (0.5, 0.40)), ((0.5, 0.40), (0.36, 0.52)), ((0.5, 0.40), (0.64, 0.52))):
        p, q = (a[0] * S, a[1] * S), (b[0] * S, b[1] * S)
        d.line([p, q], fill=WHITE, width=wd)
        for c in (p, q):
            d.ellipse((c[0] - wd / 2, c[1] - wd / 2, c[0] + wd / 2, c[1] + wd / 2), fill=WHITE)
    img = Image.alpha_composite(img, tube)
    img = img.resize((size, size), Image.LANCZOS)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path)
    return path


if __name__ == "__main__":
    print(make("assets/icon.png"))
