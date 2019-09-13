import torch
from torch import nn
import numpy as np
from torchvision.utils import save_image
from .gan import GAN as RegularGAN
from utils import bce_loss

class GAN(RegularGAN):

    def __init__(self, *args, **kwargs):
        super(GAN, self).__init__(*args, **kwargs)
        self.p = 0.5

    def sample_z(self, bs, seed=None):
        """Return a sample z ~ p(z)"""
        if seed is not None:
            rnd_state = np.random.RandomState(seed)
            z = torch.from_numpy(
                rnd_state.binomial(1, self.p, size=(bs, self.z_dim))
            ).float()
        else:
            z = torch.from_numpy(
                np.random.binomial(1, self.p, size=(bs, self.z_dim))
            ).float()
        if self.use_cuda:
            z = z.cuda()
        return z
