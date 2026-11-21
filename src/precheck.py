import re
import subprocess
from pathlib import Path

NEED = {"torch", "torchvision", "numpy", "pillow", "scikit-learn", "grad-cam", "matplotlib", "gradio", "pytest"}
BAD_END = (".pt", ".pth", ".zip")
BAD_START = ("data/", ".venv/", "space/")


def checks(root="."):
    r = Path(root)
    out = []

    def add(name, ok, msg=""):
        out.append((name, bool(ok), msg))

    rd = r / "README.md"
    t = rd.read_text(encoding="utf-8-sig") if rd.exists() else ""
    add("README exists", t)
    add("README has no TBD", t and "TBD" not in t, f"{t.count('TBD')} left")
    add("README has no NaN", t and "is nan" not in t)
    g = r / "outputs" / "demo.gif"
    add("outputs/demo.gif exists", g.exists())
    add("outputs/demo.gif under 10 MB", g.exists() and g.stat().st_size < 10_000_000)
    for f in ("roc.png", "calib.png", "results.json", "results_baseline.json", "ablation.json"):
        add(f"outputs/{f} exists", (r / "outputs" / f).exists())
    q = r / "requirements.txt"
    names = set()
    if q.exists():
        for ln in q.read_text(encoding="utf-8-sig").splitlines():
            names.add(re.split(r"[=<>~!\[; ]", ln.strip().lower())[0])
    add("requirements.txt lists all packages", NEED <= names, ", ".join(sorted(NEED - names)))
    try:
        files = subprocess.run(["git", "ls-files"], cwd=r, capture_output=True, text=True, check=True).stdout.splitlines()
        bad = [f for f in files if f.endswith(BAD_END) or f.startswith(BAD_START)]
        add("git: no weights/data/venv tracked", not bad, ", ".join(bad[:3]))
    except Exception:
        add("git: no weights/data/venv tracked", False, "git unavailable or not a repo")
    return out


if __name__ == "__main__":
    import sys

    n = 0
    for name, ok, msg in checks():
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {msg}".rstrip())
        n += not ok
    sys.exit(1 if n else 0)
