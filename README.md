# Chest X-ray Pneumonia Classifier with Grad-CAM

ResNet-18 fine-tuned to classify chest X-rays as NORMAL or PNEUMONIA, with Grad-CAM heatmaps showing which regions drove each prediction. Research demo only. Not a medical device and not for diagnosis.

Live demo: TBD (Hugging Face Space link)

![demo](outputs/demo.gif)

## Data

Kaggle "Chest X-Ray Images (Pneumonia)" (Kermany et al., pediatric chest X-rays from Guangzhou Women and Children's Medical Center). The original train/val/test folders were pooled (5,856 images) and re-split 70/15/15 (4,077 / 901 / 878).

Patient IDs are parsed from filenames and splits are grouped by patient, so no patient appears in more than one set. This is a filename-based heuristic and is not verified against metadata.

## Method

- Model: ResNet-18 (ImageNet weights), 2-class head
- Loss: cross-entropy with class weights (NORMAL 1.92, PNEUMONIA 0.68) to offset imbalance
- Training: AdamW, lr 1e-4, TBD epochs, best checkpoint by validation AUROC
- Threshold: chosen on the validation set (Youden's J), never on test (threshold 0.010)

## Results (test set)

| Metric | Value |
|---|---|
| AUROC | 0.996 |
| Sensitivity | 0.985 |
| Specificity | 0.958 |
| Confusion matrix [[TN, FP], [FN, TP]] | [[253, 11], [9, 605]] |

ROC and calibration plots are in `outputs/`.

## Explainability

Grad-CAM on the last ResNet block (`layer4`), targeting the PNEUMONIA class. Examples are in `outputs/cam/`.

Shortcut check: mean share of Grad-CAM heat in the outer 10% border is nan (56 of 200 sampled test images above 0.5; a uniform map would give 0.36). TBD (one sentence on what the heatmaps show).

Missed pneumonia cases (`outputs/fn/`): TBD (what they have in common).

## Limitations

- Single source, pediatric patients from one hospital, so results will not transfer to adults or other scanners without external validation
- Patient IDs come from filenames, not metadata
- Image-level labels with no clinician review of this model's outputs
- Grad-CAM shows where the model looked, not whether its reasoning is clinically valid
- Not a clinical tool

## Reproduce

    pip install -r requirements.txt
    python -m src.train
    python -m src.evaluate
    python -m src.plots
    python -m src.gradcam
    python -m src.shortcut
    python app.py
    pytest -q

## Credits

Dataset: Kermany et al., Cell 2018, via Kaggle (Paul Mooney's upload). Grad-CAM implementation: pytorch-grad-cam.