import torch, torchvision

def test_imports():
    import src
    assert torch.__version__ and torchvision.__version__

def test_tensor():
    assert torch.zeros(1, 3, 224, 224).shape == (1, 3, 224, 224)

def test_resnet18_forward():
    m = torchvision.models.resnet18(weights=None)
    assert m(torch.zeros(1, 3, 224, 224)).shape == (1, 1000)