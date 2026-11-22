import io

import numpy as np
import pytest
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from src.medimg import EXTS, load_image


def grad():
    return np.tile(np.linspace(0, 4095, 64), (32, 1))


def dicom(a, photo="MONOCHROME2", modality="DX", **kw):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = Dataset()
    ds.file_meta = meta
    ds.preamble = b"\0" * 128
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.Modality = modality
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photo
    ds.Rows, ds.Columns = a.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    for k, v in kw.items():
        setattr(ds, k, v)
    ds.add_new(0x7FE00010, "OW", a.astype(np.uint16).tobytes())
    b = io.BytesIO()
    ds.save_as(b)
    return b.getvalue()


def left_right(img):
    a = np.asarray(img)[..., 0].astype(float)
    return a[:, :8].mean(), a[:, -8:].mean()


def test_dicom():
    img, note = load_image(dicom(grad()), "x.dcm")
    assert img.size == (64, 32) and img.mode == "RGB" and note is None
    left, right = left_right(img)
    assert left < 50 and right > 200
    assert load_image(dicom(grad()), "scan")[0].size == (64, 32)


def test_monochrome1():
    l2, r2 = left_right(load_image(dicom(grad()), "a.dcm")[0])
    l1, r1 = left_right(load_image(dicom(grad(), photo="MONOCHROME1"), "a.dcm")[0])
    assert l2 < r2 and l1 > r1


def test_modality_note():
    _, note = load_image(dicom(grad(), modality="CT"), "a.dcm")
    assert note and "CT" in note


def test_multiframe():
    with pytest.raises(ValueError):
        load_image(dicom(grad(), NumberOfFrames=2), "a.dcm")


@pytest.mark.parametrize("fmt,ext", [("PNG", "png"), ("JPEG", "jpg"), ("WEBP", "webp"), ("BMP", "bmp"), ("TIFF", "tif")])
def test_standard(fmt, ext):
    a = np.random.randint(0, 255, (40, 50), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format=fmt)
    img, note = load_image(buf.getvalue(), "x." + ext)
    assert img.size == (50, 40) and img.mode == "RGB" and note is None


def test_16bit_png():
    a = np.tile(np.linspace(0, 65535, 64), (32, 1)).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    img, note = load_image(buf.getvalue(), "x.png")
    left, right = left_right(img)
    assert note is None and left < 50 and right > 200


def test_bad_file():
    with pytest.raises(ValueError):
        load_image(b"not an image", "x.png")


def test_exts():
    assert {"dcm", "dicom", "tif", "tiff", "bmp", "png", "jpg", "jpeg", "webp"} <= set(EXTS)
