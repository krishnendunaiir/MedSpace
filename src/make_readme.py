import json
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def render(b, t, a, n, gif=False):
    bl, tr = a["baseline"], a["trimmed"]
    rows = "\n".join(f"| {k} | {bl[k]:.3f} | {tr[k]:.3f} |" for k in bl)
    heat = n.get("heatmap_note", "TBD")
    missed = n.get("missed_note", "TBD")
    lines = [
        "# Chest X-ray Pneumonia Classifier with Grad-CAM",
        "",
        "ResNet-18 fine-tuned to classify chest X-rays as NORMAL or PNEUMONIA, with Grad-CAM heatmaps and ablation tests that check what the model relies on. Research demo only. Not a medical device and not for diagnosis.",
        "",
        *([f"Live demo: {n['space_url']}", ""] if n.get("space_url") else []),
        *(["![demo](outputs/demo.gif)", ""] if gif else []),
        "## Data",
        "",
        "Kaggle \"Chest X-Ray Images (Pneumonia)\" (Kermany et al., pediatric chest X-rays from Guangzhou Women and Children's Medical Center). The original train/val/test folders were pooled (5,856 images) and re-split 70/15/15 (4,077 / 901 / 878).",
        "",
        "Patient IDs are parsed from filenames and splits are grouped by patient, so no patient appears in more than one set. This is a filename-based heuristic and is not verified against metadata.",
        "",
        "## Method",
        "",
        "- Model: ResNet-18 (ImageNet weights), 2-class head, class-weighted cross-entropy (NORMAL 1.92, PNEUMONIA 0.68)",
        "- Training: AdamW, lr 1e-4, 5 epochs, best checkpoint by validation AUROC, trained on a Kaggle GPU",
        "- Two models: a baseline on full images, and a final model trained and evaluated with the outer 10% of each image cropped off",
        "- Decision threshold chosen on the validation set (Youden's J), never on the test set",
        "",
        "## Results",
        "",
        "Test set (878 images, patient-disjoint from training):",
        "",
        "| Metric | Baseline | Trimmed (final) |",
        "|---|---|---|",
        f"| ROC AUC | {b['auroc']:.3f} | {t['auroc']:.3f} |",
        f"| Sensitivity | {b['sensitivity']:.3f} | {t['sensitivity']:.3f} |",
        f"| Specificity | {b['specificity']:.3f} | {t['specificity']:.3f} |",
        f"| Confusion matrix [[TN, FP], [FN, TP]] | {b['cm']} | {t['cm']} |",
        f"| Threshold (chosen on val) | {b['threshold']:.4f} | {t['threshold']:.4f} |",
        "",
        "ROC and calibration plots for the final model are in `outputs/`. Predicted probabilities were not calibrated or validated as risk estimates.",
        "",
        "## Explainability",
        "",
        "Grad-CAM on the last ResNet block (`layer4`), targeting the PNEUMONIA class. Examples are in `outputs/cam/`.",
        "",
        "Qualitative review of the baseline's overlays found heat sometimes on the shoulders or spread across the whole image. Because that suggested non-anatomical cues, I tested it directly. Each test image was blanked in part and the model re-scored (test AUROC; 0.5 is chance):",
        "",
        "| Input visible to the model | Baseline | Trimmed |",
        "|---|---|---|",
        rows,
        "",
        f"- Trimming the outer 10% changed AUROC on unmodified test images from {bl['normal']:.3f} to {tr['normal']:.3f}.",
        f"- With only the outer 3% frame visible, AUROC went from {bl['frame 3%']:.3f} (baseline) to {tr['frame 3%']:.3f} (trimmed). A model reading lung findings should not separate classes from the extreme edge, so the baseline's score points to non-anatomical cues.",
        f"- Mean share of Grad-CAM heat in the outer 10% frame: {b['border_share']:.2f} baseline vs {t['border_share']:.2f} trimmed (a uniform map gives 0.36). Maps with no heat ({b['n_empty']} and {t['n_empty']} of 200 sampled images) are excluded.",
        "",
        f"Trimmed-model heatmaps: {heat}",
        "",
        f"Missed pneumonia cases (`outputs/fn/`): {missed}",
        "",
        "## Limitations",
        "",
        "- Single source: pediatric patients from one hospital, so results will not transfer to adults or other scanners without external validation",
        "- The splits re-pool Kaggle's original folders, so the scores measure in-distribution performance and are probably optimistic",
        "- Patient IDs come from filenames, not metadata",
        "- The ablations are consistent with reduced edge dependence but do not prove the model reads lung findings: blanked images are unlike the training data, the frame rows cover different original pixels in the two models, and there is one split and one seed with no confidence intervals",
        "- Grad-CAM from the last block is coarse and shows where the model looked, not whether its reasoning is clinically valid",
        "- Labels are image-level with no clinician review of this model's outputs",
        "- Not a clinical tool",
        "",
        "## Reproduce",
        "",
        "    pip install -r requirements.txt",
        "    # train on a GPU (a Kaggle notebook works); baseline: trim 0, trimmed: trim 0.10",
        "    python -m src.report       # metrics, border share -> outputs/results.json",
        "    python -m src.plots",
        "    python -m src.gradcam",
        "    python -m src.compare      # baseline vs trimmed ablations",
        "    python -m src.make_readme",
        "    python app.py",
        "    pytest -q",
        "",
        "The weights and trim default to the final model; override with the CXR_WEIGHTS and CXR_TRIM environment variables.",
        "",
        "## Credits",
        "",
        "Dataset: Kermany et al., Cell 2018, via Kaggle (Paul Mooney's upload). Grad-CAM implementation: pytorch-grad-cam.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    o = Path("outputs")
    notes = load(o / "notes.json") if (o / "notes.json").exists() else {}
    text = render(load(o / "results_baseline.json"), load(o / "results.json"), load(o / "ablation.json"), notes, (o / "demo.gif").exists())
    Path("README.md").write_text(text, encoding="utf-8")
