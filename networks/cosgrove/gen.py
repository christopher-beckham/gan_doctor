from .core import (Generator)

def get_network(z_dim, nf):
    gen = Generator(z_dim=z_dim, nf=nf)
    return gen
