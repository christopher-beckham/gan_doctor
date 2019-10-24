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

    # If we enable spectral norm, then we
    # disable the equalised learning rate
    # trick.
    use_sn = net_kwargs.pop('use_sn', False)
    if use_sn:
        use_eql = False
    else:
        use_eql = True

    # This doesn't use spectral norm in either
    # of the networks.
    gen = Generator(latent_size=z_dim,
                    depth=6,
                    norm_layer=net_kwargs['g_norm'],
                    use_eql=use_eql)
    disc = Discriminator(depth=6,
                         feature_size=z_dim,
                         sigm=True if gan_args['loss']=='bce' else False,
                         use_sn=use_sn,
                         use_eql=use_eql)
    return gen, disc

if __name__ == '__main__':
    gen, disc = get_network(128)
    print(gen)
    print(disc)
