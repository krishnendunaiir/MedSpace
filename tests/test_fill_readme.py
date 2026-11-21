from pathlib import Path

from src.fill_readme import fill

R = {
    "auroc": 0.97,
    "sensitivity": 0.95,
    "specificity": 0.8,
    "cm": [[2, 0], [1, 1]],
    "threshold": 0.8,
    "border_share": 0.21,
    "border_over_half": 3,
    "n_border": 200,
}
GONE = [
    "| AUROC | TBD |",
    "| Sensitivity | TBD |",
    "| Specificity | TBD |",
    "| Confusion matrix [[TN, FP], [FN, TP]] | TBD |",
    "is TBD (a uniform",
]


def test_fill():
    t = "| AUROC | TBD |\nnever on test\nis TBD (a uniform map would give 0.36)."
    o = fill(t, R)
    assert "| AUROC | 0.970 |" in o and "(threshold 0.800)" in o
    assert "0.21 (3 of 200 sampled" in o
    assert fill(o, R) == o


def test_fix_nan():
    t = "is nan (56 of 200 sampled test images above 0.5; a uniform map would give 0.36)."
    assert "is 0.21 (3 of 200 sampled" in fill(t, R)


def test_real_readme():
    o = fill(Path("README.md").read_text(encoding="utf-8"), R)
    for g in GONE:
        assert g not in o, g
