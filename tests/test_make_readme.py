from src.make_readme import render

R = {"auroc": 0.99, "sensitivity": 0.9, "specificity": 0.8, "cm": [[1, 2], [3, 4]], "threshold": 0.05, "border_share": 0.4, "n_empty": 6}
T = {**R, "auroc": 0.98, "border_share": 0.3, "n_empty": 16}
K = ["normal", "frame 3%", "frame 6%", "frame 10%", "frame 15%", "center only"]
A = {"baseline": {k: 0.9 for k in K}, "trimmed": {k: 0.6 for k in K}}
A["baseline"]["normal"], A["trimmed"]["normal"] = 0.996, 0.993


def test_sections():
    t = render(R, T, A, {})
    assert t.startswith("# Chest X-ray")
    for s in ("## Data", "## Method", "## Results", "## Explainability", "## Limitations", "## Reproduce", "## Credits"):
        assert s in t, s
    assert "not for diagnosis" in t
    assert "demo.gif" not in t and "Live demo" not in t
    assert "outputs/demo.gif" in render(R, T, A, {}, True)


def test_numbers():
    t = render(R, T, A, {})
    assert "| ROC AUC | 0.990 | 0.980 |" in t
    assert "from 0.996 to 0.993" in t
    assert "0.40 baseline vs 0.30 trimmed" in t and "(6 and 16 of 200" in t
    assert "| frame 3% | 0.900 | 0.600 |" in t


def test_notes():
    assert render(R, T, A, {}).count("TBD") == 2
    n = {"heatmap_note": "h", "missed_note": "m"}
    t = render(R, T, A, n)
    assert "TBD" not in t and render(R, T, A, n) == t
