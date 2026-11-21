from pathlib import Path

import pytest
import torch

from src.model import make


@pytest.mark.parametrize("f", ["best.pt", "best_trim.pt"])
def test_weights(f):
    if not Path(f).exists():
        pytest.skip(f"{f} missing")
    make(False).load_state_dict(torch.load(f, map_location="cpu"))
