# Chest X-ray Pneumonia Classifier with Grad-CAM

ResNet-18 fine-tuned to classify chest X-rays as NORMAL or PNEUMONIA, with Grad-CAM heatmaps and ablation tests that check what the model relies on. Research demo only. Not a medical device and not for diagnosis.

Live demo: TBD

![demo](outputs/demo.gif)

## Data

Kaggle "Chest X-Ray Images (Pneumonia)" (Kermany et al., pediatric chest X-rays from Guangzhou Women and Children's Medical Center). The original train/val/test folders were pooled (5,856 images) and re-split 70/15/15 (4,077 / 901 / 878).

Patient IDs are parsed from filenames and splits are grouped by patient, so no patient appears in more than one set. This is a filename-based heuristic and is not verified against metadata.

## Method

- Model: ResNet-18 (ImageNet weights), 2-class head, class-weighted cross-entropy (NORMAL 1.92, PNEUMONIA 0.68)
- Training: AdamW, lr 1e-4, 5 epochs, best checkpoint by validation AUROC, trained on a Kaggle GPU
- Two models: a baseline on full images, and a final model trained and evaluated with the outer 10% of each image cropped off
- Decision threshold chosen on the validation set (Youden's J), never on the test set

## Results

Test set (878 images, patient-disjoint from training):

| Metric | Baseline | Trimmed (final) |
|---|---|---|
| ROC AUC | 0.996 | 0.993 |
| Sensitivity | 0.985 | 0.966 |
| Specificity | 0.958 | 0.955 |
| Confusion matrix [[TN, FP], [FN, TP]] | [[253, 11], [9, 605]] | [[252, 12], [21, 593]] |
| Threshold (chosen on val) | 0.0104 | 0.1085 |

ROC and calibration plots for the final model are in `outputs/`. Predicted probabilities were not calibrated or validated as risk estimates.

## Explainability

Grad-CAM on the last ResNet block (`layer4`), targeting the PNEUMONIA class. Examples are in `outputs/cam/`.

Qualitative review of the baseline's overlays found heat sometimes on the shoulders or spread across the whole image. Because that suggested non-anatomical cues, I tested it directly. Each test image was blanked in part and the model re-scored (test AUROC; 0.5 is chance):

| Input visible to the model | Baseline | Trimmed |
|---|---|---|
| normal | 0.996 | 0.993 |
| frame 3% | 0.858 | 0.570 |
| frame 6% | 0.912 | 0.571 |
| frame 10% | 0.949 | 0.829 |
| frame 15% | 0.965 | 0.928 |
| center only (outer 25% blanked) | 0.894 | 0.687 |

- Trimming the outer 10% changed AUROC on unmodified test images from 0.996 to 0.993.
- With only the outer 3% frame visible, AUROC went from 0.858 (baseline) to 0.570 (trimmed). A model reading lung findings should not separate classes from the extreme edge, so the baseline's score points to non-anatomical cues.
- Mean share of Grad-CAM heat in the outer 10% frame: 0.43 baseline vs 0.37 trimmed (a uniform map gives 0.36). Maps with no heat (6 and 16 of 200 sampled images) are excluded.

Trimmed-model heatmaps: TBD

Missed pneumonia cases (`outputs/fn/`): TBD

## Limitations

- Single source: pediatric patients from one hospital, so results will not transfer to adults or other scanners without external validation
- The splits re-pool Kaggle's original folders, so the scores measure in-distribution performance and are probably optimistic
- Patient IDs come from filenames, not metadata
- The ablations are consistent with reduced edge dependence but do not prove the model reads lung findings: blanked images are unlike the training data, the frame rows cover different original pixels in the two models, and there is one split and one seed with no confidence intervals
- Grad-CAM from the last block is coarse and shows where the model looked, not whether its reasoning is clinically valid
- Labels are image-level with no clinician review of this model's outputs
- Not a clinical tool

## Reproduce

    pip install -r requirements.txt
    # train on a GPU (a Kaggle notebook works); baseline: trim 0, trimmed: trim 0.10
    python -m src.report       # metrics, border share -> outputs/results.json
    python -m src.plots
    python -m src.gradcam
    python -m src.compare      # baseline vs trimmed ablations
    python -m src.make_readme
    python app.py
    pytest -q

The weights and trim default to the final model; override with the CXR_WEIGHTS and CXR_TRIM environment variables.

## Credits

Dataset: Kermany et al., Cell 2018, via Kaggle (Paul Mooney's upload). Grad-CAM implementation: pytorch-grad-cam.
