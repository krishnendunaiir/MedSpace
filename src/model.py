import torch
from torch import nn
from torchvision import models


def make(pretrained=True):
    m = models.resnet18(weights="DEFAULT" if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def class_weights(items):
    n = torch.bincount(torch.tensor([y for _, y in items]), minlength=2).float()
    return n.sum() / (2 * n)
