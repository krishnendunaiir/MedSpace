import glob
import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms as T

from src.split import split

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLASSES = ["NORMAL", "PNEUMONIA"]
EXT = (".jpeg", ".jpg", ".png")


def tf(train):
    aug = [T.RandomRotation(10), T.ColorJitter(0.2, 0.2)] if train else []
    return T.Compose([T.Grayscale(3), T.Resize((224, 224)), *aug, T.ToTensor(), T.Normalize(MEAN, STD)])


def folder(root, train=False):
    return datasets.ImageFolder(root, tf(train))


def loader(ds, bs=32, shuffle=False):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0)


def gather(root):
    return sorted(
        (p, y)
        for y, c in enumerate(CLASSES)
        for p in glob.glob(os.path.join(root, "*", c, "*"))
        if p.lower().endswith(EXT) and not os.path.basename(p).startswith(".")
    )


class CXR(Dataset):
    def __init__(self, items, train=False):
        self.items, self.t = items, tf(train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.t(Image.open(p)), y


def build(root, seed=42):
    it = gather(root)
    parts = split([p for p, _ in it], seed)
    return [CXR([it[i] for i in ix], k == 0) for k, ix in enumerate(parts)]
