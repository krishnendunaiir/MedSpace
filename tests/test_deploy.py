import gradio as gr

from deploy import FILES, make_space


def test_make_space(tmp_path):
    w = tmp_path / "best_trim.pt"
    w.write_bytes(b"x")
    out = make_space(str(tmp_path / "space"), str(w))
    assert "src/cfg.py" in FILES
    for f in [*FILES, "best_trim.pt", "requirements.txt", "README.md"]:
        assert (out / f).is_file(), f
    r = (out / "README.md").read_text(encoding="utf-8")
    assert r.startswith("---\ntitle:") and f"sdk_version: {gr.__version__}" in r
    assert "gradio" in (out / "requirements.txt").read_text()
