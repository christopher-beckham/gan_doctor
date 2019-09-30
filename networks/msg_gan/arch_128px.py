from .MSG_GAN.GAN import (Generator,
                          Discriminator)

def get_network(z_dim):
    gen = Generator(latent_size=z_dim, depth=6, use_spectral_norm=False, norm_layer='batch') # 128px
    disc = Discriminator(depth=6, feature_size=z_dim, sigm=True)
    return gen, disc

if __name__ == '__main__':
    gen, disc = get_network(128)
    print(gen)
    print(disc)
