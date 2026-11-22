import base64
import os
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_paste_button import paste_image_button

from src.app import label, load, run_stages
from src.calibrate import load_cal
from src.cfg import WEIGHTS
from src.medimg import EXTS, load_image
from src.weights import ensure

ICON = Path(__file__).resolve().parent / "assets" / "icon.png"
st.set_page_config(page_title="Chest X-ray classifier", page_icon=Image.open(ICON), layout="centered")

MODES = ["Upload / drag & drop", "Webcam", "Paste from clipboard"]
CAL = load_cal()


def weights_url():
    u = os.environ.get("CXR_WEIGHTS_URL")
    if u:
        return u
    try:
        return st.secrets.get("CXR_WEIGHTS_URL")
    except Exception:
        return None


@st.cache_resource
def model():
    return load(ensure(WEIGHTS, weights_url()))


def to_image(f):
    try:
        img, note = load_image(f.getvalue(), f.name)
    except Exception as e:
        st.error(f"Could not read that file: {e}")
        return None
    if note:
        st.warning(note)
    return img


def get_image(mode):
    if mode == MODES[0]:
        f = st.file_uploader("Drag an X-ray here, or click to browse (DICOM, PNG, JPG, WebP, TIFF, BMP)", type=EXTS)
        return to_image(f) if f else None
    if mode == MODES[1]:
        f = st.camera_input("Take a photo")
        return to_image(f) if f else None
    r = paste_image_button("Paste image from clipboard", key="paste")
    return r.image_data.convert("RGB") if r.image_data is not None else None


LOGO = base64.b64encode(ICON.read_bytes()).decode()
st.markdown(
    f"<div style='text-align:center'><img src='data:image/png;base64,{LOGO}' width='88'>"
    "<h1 style='margin:0.1em 0 0'>Chest X-ray classifier</h1></div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center'>Add a chest X-ray to see what the model predicts and where it looked.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center'><small>Research demo only. Not a medical device and not for diagnosis. "
    "Do not upload real patient data.</small></p>",
    unsafe_allow_html=True,
)

with st.container(border=True):
    mode = st.radio("How do you want to add an image?", MODES, horizontal=True)
    img = get_image(mode)

if img is None:
    st.info("Waiting for an image. The model was trained on digital chest X-ray files, so photos of screens or printed films may give unreliable results.")
else:
    bar = st.progress(0.05, text="Loading model...")
    m = model()
    probs, cam = run_stages(m, img, lambda v, t: bar.progress(v, text=t), CAL)
    bar.empty()
    st.subheader(f"Model output: {label(probs, CAL)}")
    for k, v in probs.items():
        st.progress(min(max(v, 0.0), 1.0), text=f"{k}: {v:.1%}")
    if CAL:
        st.caption(
            "Scores are calibrated with a logistic fit on a validation set from this dataset, "
            f"and the label uses a tuned threshold of {CAL['threshold']:.2f}. "
            "They reflect this dataset's pneumonia prevalence and are not patient risk or a diagnosis."
        )
    else:
        st.caption("These are uncalibrated model scores, not a diagnosis.")
    c1, c2 = st.columns(2)
    c1.image(img, caption="Your image")
    c2.image(cam, caption="Grad-CAM: regions that pushed toward pneumonia")

with st.expander("About this demo"):
    st.write(
        "ResNet-18 fine-tuned on pediatric chest X-rays from one hospital (Kaggle dataset), with the outer 10% of each image cropped off. "
        "It may not transfer to adults or other scanners. Grad-CAM shows where the model looked, not whether its reasoning is clinically valid. "
        "Accepted formats: DICOM (.dcm), PNG, JPG, WebP, TIFF, BMP. For DICOM, only the pixel data is used and header fields such as patient name are not displayed or saved by this app, "
        "but the file is still uploaded to wherever the app runs, so use de-identified images."
    )

st.caption("Tip: switch between light and dark mode from the menu at the top right (Settings, then Theme). It follows your system setting by default.")
