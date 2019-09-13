from torchvision.datasets import CIFAR10
from torchvision import transforms

def get_dataset(**kwargs):
    ds = CIFAR10("./",
                 train=True,
                 transform=transforms.ToTensor(),
                 target_transform=None,
                 download=True)
    return ds
