from pathlib import Path

SECTIONS = ["## Data", "## Method", "## Results", "## Explainability", "## Limitations", "## Reproduce", "## Credits"]


def test_readme():
    t = Path("README.md").read_text(encoding="utf-8")
    assert t.startswith("# Chest X-ray")
    for s in SECTIONS:
        assert s in t, s
    assert "not for diagnosis" in t.lower()
