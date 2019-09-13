import torch
from torch import nn
import numpy as np
from torchvision.utils import save_image
from .base import GAN as Base
from utils import bce_loss

class GAN(Base):

    def __init__(self, *args, **kwargs):
        super(GAN, self).__init__(*args, **kwargs)

    def prepare_batch(self, batch):
        x = batch[0]
        if self.use_cuda:
            x = x.cuda()
        return [x]

    def sample(self, bs, seed=None):
        self._eval()
        gz = self.g(self.sample_z(bs=bs, seed=seed))
        return gz

    def sample_z(self, bs, seed=None):
        """Return a sample z ~ p(z)"""
        if seed is not None:
            rnd_state = np.random.RandomState(seed)
            z = torch.from_numpy(
                rnd_state.normal(0, 1, size=(bs, self.z_dim))
            ).float()
        else:
            z = torch.from_numpy(
                np.random.normal(0, 1, size=(bs, self.z_dim))
            ).float()
        if self.use_cuda:
            z = z.cuda()
        return z

    def train_on_instance(self, z, x, **kwargs):
        self._train()
        # Train the generator.
        self.optim['g'].zero_grad()
        fake = self.g(z)
        d_fake = self.d(fake)
        gen_loss = bce_loss(d_fake, 1)
        if (kwargs['iter']-1) % self.update_g_every == 0:
            gen_loss.backward()
            self.optim['g'].step()
        # Train the discriminator.
        self.optim['d'].zero_grad()
        d_fake = self.d(fake.detach())
        d_real = self.d(x)
        d_loss = bce_loss(d_real, 1) + bce_loss(d_fake, 0)
        d_loss.backward()
        self.optim['d'].step()
        losses = {
            'g_loss': gen_loss.data.item(),
            'd_loss': d_loss.data.item() / 2.
        }
        outputs = {
            'x': x.detach(),
            'gz': fake.detach(),
        }
        return losses, outputs
