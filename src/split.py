import re

from sklearn.model_selection import GroupShuffleSplit


def pid(path):
    n = path.replace("\\", "/").split("/")[-1]
    m = re.match(r"(person\d+)", n) or re.search(r"IM-(\d+)", n)
    return m.group(1) if m else n


def split(paths, seed=42, val=0.15, test=0.15):
    g = [pid(p) for p in paths]
    idx = list(range(len(paths)))
    a, te = next(GroupShuffleSplit(1, test_size=test, random_state=seed).split(idx, groups=g))
    tr, va = next(GroupShuffleSplit(1, test_size=val / (1 - test), random_state=seed).split(a, groups=[g[i] for i in a]))
    return a[tr].tolist(), a[va].tolist(), te.tolist()
