# ResNet generator and discriminator
from torch import nn
import torch.nn.functional as F

from .spec_norm_old import SpectralNorm
#from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
import numpy as np

channels = 3

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.spec_norm(self.conv1),
                nn.ReLU(),
                self.spec_norm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.spec_norm(self.conv1),
                nn.ReLU(),
                self.spec_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                self.spec_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.spec_norm(self.conv1),
            nn.ReLU(),
            self.spec_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.spec_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class Generator(nn.Module):
    def __init__(self, z_dim, GEN_SIZE=128):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh())

        self.GEN_SIZE = GEN_SIZE

    def forward(self, z):
        return self.model(self.dense(z).view(-1, self.GEN_SIZE, 4, 4))

class Discriminator(nn.Module):
    def __init__(self, spec_norm=False, sigmoid=False, DISC_SIZE=128):
        super(Discriminator, self).__init__()

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x : x

        self._init = FirstResBlockDiscriminator(channels, DISC_SIZE,
                                                stride=2, spec_norm=spec_norm)
        self._init2 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,
                                             stride=2, spec_norm=spec_norm)
        self.model = nn.Sequential(
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,
                                      spec_norm=spec_norm),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,
                                      spec_norm=spec_norm),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = self.spec_norm(self.fc)
        if sigmoid:
            self.sigm = nn.Sigmoid()
        self.sigmoid = sigmoid
        self.DISC_SIZE = DISC_SIZE

    def forward(self, x):
        """
        Return a tuple of intermediate states, and also
          the final output.
        """
        init = self._init(x)
        init2 = self._init2(init)
        return self.partial_forward(init2, 1)

    def partial_forward(self, hs, idx):
        """
        Compute the output of the discriminator, given
          either the result of the first or second layer.
        """
        assert idx in [0,1]
        if idx == 0:
            # hs == the result of self._init
            result = self.fc(self.model(self._init2(hs)).view(-1,self.DISC_SIZE))
        else:
            # hs == the result of self._init2
            result = self.fc(self.model(hs).view(-1,self.DISC_SIZE))
        if self.sigmoid:
            return self.sigm(result)
        else:
            return result
        
