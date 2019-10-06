from .MSG_GAN.GAN import (Generator,
                          Discriminator)

def get_network(z_dim, gan_args, **kwargs):
    """
    z_dim: z dimension
    gan_args: whatever args were passed to the gan class
    kwargs: extra args you can pass
    """

    print("gan_args: %s" % gan_args)
    print("kwargs: %s" % kwargs)

    gen = Generator(latent_size=z_dim, depth=6,
                    use_spectral_norm=kwargs['use_sn'],
                    norm_layer=kwargs['norm'])
    disc = Discriminator(depth=6,
                         feature_size=z_dim,
                         sigm=True if gan_args['loss']=='bce' else False)
    return gen, disc

if __name__ == '__main__':
    gen, disc = get_network(128)
    print(gen)
    print(disc)
