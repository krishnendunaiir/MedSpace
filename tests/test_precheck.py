from src.precheck import checks

GIT = "git: no weights/data/venv tracked"


def make(p, readme="# x\nall good\n"):
    (p / "outputs").mkdir()
    for f in ("roc.png", "calib.png", "results.json", "results_baseline.json", "ablation.json"):
        (p / "outputs" / f).write_text("x")
    (p / "outputs" / "demo.gif").write_bytes(b"GIF89a")
    (p / "README.md").write_text(readme, encoding="utf-8")
    (p / "requirements.txt").write_text("torch\ntorchvision\nnumpy\npillow\nscikit-learn\ngrad-cam\nmatplotlib\ngradio\npytest\n")


def res(p):
    return {n: ok for n, ok, _ in checks(str(p))}


def test_pass(tmp_path):
    make(tmp_path)
    assert all(v for k, v in res(tmp_path).items() if k != GIT)


def test_fail(tmp_path):
    make(tmp_path, "# x\nTBD\nis nan (1)\n")
    (tmp_path / "outputs" / "demo.gif").unlink()
    (tmp_path / "requirements.txt").write_text("torch\npytestmatplotlib\n")
    r = res(tmp_path)
    assert not r["README has no TBD"] and not r["README has no NaN"]
    assert not r["outputs/demo.gif exists"] and not r["requirements.txt lists all packages"]


def test_missing_readme(tmp_path):
    (tmp_path / "outputs").mkdir()
    r = res(tmp_path)
    assert not r["README exists"] and not r["README has no TBD"]
