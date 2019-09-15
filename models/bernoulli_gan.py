import torch
from torch import nn
import numpy as np
from torchvision.utils import save_image
from .gan import GAN as RegularGAN
from utils import bce_loss

class GAN(RegularGAN):

    REQUIRED_ARGS = {
        'p': {
            'desc': 'Bernoulli parameter p',
            'default': 0.5
        },
        'center': {
            'desc': 'Center sampled values to lie in range {-1, 1}, rather than {0, 1}',
            'default': False
        }
    }

    def __init__(self, *args, **kwargs):
        self._validate(kwargs)
        super(GAN, self).__init__(*args, **kwargs)

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
        if self.center:
            z = (z - 0.5) / 0.5
        if self.use_cuda:
            z = z.cuda()
        return z
