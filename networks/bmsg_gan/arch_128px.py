from .MSG_GAN.GAN import (Generator,
                          Discriminator)

def get_network(z_dim, gan_args, **net_kwargs):
    """
    z_dim: z dimension
    gan_args: whatever args were passed to the gan class
    kwargs: extra args you can pass
    """

    print("gan_args: %s" % gan_args)
    print("kwargs: %s" % net_kwargs)

    # This doesn't use spectral norm in either
    # of the networks.
    gen = Generator(latent_size=z_dim, depth=6,
                    norm_layer=net_kwargs['norm'])
    use_sn_d = net_kwargs.pop('use_sn_d', False)
    disc = Discriminator(depth=6,
                         feature_size=z_dim,
                         sigm=True if gan_args['loss']=='bce' else False,
                         use_sn=use_sn_d,
                         use_eql=False if use_sn_d else True) # if sn is off, use eql
    return gen, disc

if __name__ == '__main__':
    gen, disc = get_network(128)
    print(gen)
    print(disc)
