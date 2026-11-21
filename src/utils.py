import random, numpy as np, torch

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"
