import json
import re
from pathlib import Path

BORDER = re.compile(r"\bis (?:TBD|nan|\d\.\d+) \([^)]*\)\.")


def row(t, k, v):
    return re.sub(rf"^\| {re.escape(k)} \| .*\|$", lambda _: f"| {k} | {v} |", t, count=1, flags=re.M)


def fill(t, r):
    t = row(t, "AUROC", f"{r['auroc']:.3f}")
    t = row(t, "Sensitivity", f"{r['sensitivity']:.3f}")
    t = row(t, "Specificity", f"{r['specificity']:.3f}")
    t = row(t, "Confusion matrix [[TN, FP], [FN, TP]]", str(r["cm"]))
    msg = f"is {r['border_share']:.2f} ({r['border_over_half']} of {r['n_border']} sampled test images above 0.5; a uniform map would give 0.36)."
    t = BORDER.sub(lambda _: msg, t, count=1)
    th = f"(threshold {r['threshold']:.3f})"
    if "(threshold " in t:
        t = re.sub(r"\(threshold [^)]*\)", lambda _: th, t, count=1)
    else:
        t = t.replace("never on test", f"never on test {th}", 1)
    return t


if __name__ == "__main__":
    r = json.loads(Path("outputs/results.json").read_text(encoding="utf-8"))
    p = Path("README.md")
    p.write_text(fill(p.read_text(encoding="utf-8"), r), encoding="utf-8")
