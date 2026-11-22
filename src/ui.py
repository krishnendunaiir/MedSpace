import gradio as gr

from src.app import predict

NOTE = "Research demo only. Not a medical device and not for diagnosis. Do not upload real patient data."


def build_ui(m):
    def run(img):
        if img is None:
            return None, None
        return predict(m, img)

    return gr.Interface(
        run,
        gr.Image(type="numpy", label="Chest X-ray"),
        [gr.Label(label="Prediction"), gr.Image(label="Grad-CAM (pneumonia evidence)")],
        title="Chest X-ray pneumonia classifier",
        description=NOTE,
        flagging_mode="never",
    )
