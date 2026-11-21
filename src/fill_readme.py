import json
from pathlib import Path


def fill(t, r):
    s = (
        ("| AUROC | TBD |", f"| AUROC | {r['auroc']:.3f} |"),
        ("| Sensitivity | TBD |", f"| Sensitivity | {r['sensitivity']:.3f} |"),
        ("| Specificity | TBD |", f"| Specificity | {r['specificity']:.3f} |"),
        ("| Confusion matrix [[TN, FP], [FN, TP]] | TBD |", f"| Confusion matrix [[TN, FP], [FN, TP]] | {r['cm']} |"),
        (
            "is TBD (a uniform map would give 0.36).",
            f"is {r['border_share']:.2f} ({r['border_over_half']} of {r['n_border']} sampled test images above 0.5; a uniform map would give 0.36).",
        ),
    )
    for a, b in s:
        t = t.replace(a, b, 1)
    if "(threshold " not in t:
        t = t.replace("never on test", f"never on test (threshold {r['threshold']:.3f})", 1)
    return t


if __name__ == "__main__":
    r = json.loads(Path("outputs/results.json").read_text(encoding="utf-8"))
    p = Path("README.md")
    p.write_text(fill(p.read_text(encoding="utf-8"), r), encoding="utf-8")
