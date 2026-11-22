import io
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

EXTS = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "dcm", "dicom"]
CHEST_MODALITIES = {"", "CR", "DX", "RG"}


def to_u8(a, lo=None, hi=None):
    a = np.asarray(a, dtype=np.float32)
    if lo is None or hi is None:
        lo, hi = np.percentile(a, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)


def first(v):
    try:
        return float(v)
    except TypeError:
        return float(v[0])


def read_dicom(data):
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    if int(ds.get("NumberOfFrames", 1) or 1) > 1:
        raise ValueError("Multi-frame DICOM (volumes or cine) is not supported. Please use a single 2D X-ray image.")
    try:
        a = ds.pixel_array.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Could not decode the DICOM pixel data ({type(e).__name__}). It may use an unsupported compression format.") from e
    if a.ndim == 3 and a.shape[-1] == 3:
        a = a.mean(-1)
    if a.ndim != 2:
        raise ValueError("Unsupported DICOM image layout.")
    a = a * first(ds.get("RescaleSlope", 1)) + first(ds.get("RescaleIntercept", 0))
    lo = hi = None
    wc, ww = ds.get("WindowCenter"), ds.get("WindowWidth")
    if wc is not None and ww is not None and first(ww) > 0:
        c, w = first(wc), first(ww)
        lo, hi = c - w / 2, c + w / 2
    u = to_u8(a, lo, hi)
    if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
        u = 255 - u
    mod = str(ds.get("Modality", ""))
    note = None
    if mod not in CHEST_MODALITIES:
        note = f"This DICOM's modality is {mod}, but the model was trained on chest X-rays (CR/DX). Results will not be meaningful."
    return Image.fromarray(u).convert("RGB"), note


def load_image(data, name=""):
    if Path(name).suffix.lower() in {".dcm", ".dicom"} or data[128:132] == b"DICM":
        return read_dicom(data)
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Unrecognized or corrupted image file.") from e
    if im.mode.startswith("I") or im.mode == "F":
        return Image.fromarray(to_u8(np.asarray(im))).convert("RGB"), None
    return im.convert("RGB"), None
