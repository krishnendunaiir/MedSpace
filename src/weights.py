import urllib.request
from pathlib import Path


def ensure(path, url=None):
    p = Path(path)
    if p.exists():
        return str(p)
    if not url:
        raise FileNotFoundError(f"{path} is missing and no download URL is set")
    tmp = p.with_name(p.name + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(p)
    return str(p)
