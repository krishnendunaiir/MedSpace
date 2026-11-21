import os

WEIGHTS = os.environ.get("CXR_WEIGHTS", "best_trim.pt")
TRIM = float(os.environ.get("CXR_TRIM", "0.10"))
