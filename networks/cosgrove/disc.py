from .core import (Discriminator)

def get_network(nf,
                spec_norm=True,
                sigmoid=True):
    disc = Discriminator(nf=nf,
                         spec_norm=spec_norm,
                         sigmoid=sigmoid)
    return disc
