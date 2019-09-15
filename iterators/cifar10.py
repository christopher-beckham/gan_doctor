from torchvision.datasets import CIFAR10
from torchvision import transforms

NORMALISE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def get_dataset(**kwargs):
    ds = CIFAR10("./",
                 train=True,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     NORMALISE
                 ]),
                 target_transform=None,
                 download=True)
    return ds
