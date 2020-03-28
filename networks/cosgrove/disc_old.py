from .core_old import (Discriminator)

def get_network(nf,
                spec_norm=True,
                sigmoid=True):
    disc = Discriminator(spec_norm=spec_norm,
                         sigmoid=sigmoid)
    return disc
