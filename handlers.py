import torch
import pickle
import os
import numpy as np
from torchvision.utils import save_image
from utils import (compute_fid, compute_is)

def is_handler(gan,
               n_samples,
               batch_size,
               eval_every=5):
    '''Callback for computing Inception Score.

    Params
    ------

    '''
    def _is_handler(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1 and \
           kwargs['mode'] == 'train' and \
           kwargs['epoch'] % eval_every == 0:
            gan._eval()
            with torch.no_grad():
                return compute_is(n_samples=n_samples,
                                  gan=gan,
                                  batch_size=batch_size)
        return {}
    return _is_handler

def fid_handler(gan,
                cls,
                loader,
                n_samples,
                batch_size=None,
                eval_every=5):
    '''Callback for computing FID score.

    Params
    ------

    '''

    #if batch_size is None:
    #    batch_size = loader.batch_size
    # Extract the loader here...

    # Extract `n_samples` from the training set
    # here.
    train_samples = []
    counter = 0
    for b, (x_batch, _) in enumerate(loader):
        train_samples.append(x_batch.numpy())
        counter += len(x_batch)
        if counter >= n_samples:
            break
    train_samples = np.vstack(train_samples)[0:n_samples]

    def _fid_handler(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1 and \
           kwargs['mode'] == 'train' and \
           kwargs['epoch'] % eval_every == 0:
            gan._eval()
            with torch.no_grad():
                return compute_fid(train_samples=train_samples,
                                   gan=gan,
                                   batch_size=batch_size,
                                   cls=cls)
        return {}
    return _fid_handler

def dump_img_handler(gan,
                     dest_dir,
                     batch_size,
                     eval_every=1):
    def _dump_img_handler(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1 and \
           kwargs['mode'] == 'train' and \
           kwargs['epoch'] % eval_every == 0:
            gan._eval()
            with torch.no_grad():
                gen_imgs = gan.sample(batch_size)
                if type(gen_imgs) == list:
                    # Special case to handle MSG-GAN
                    for j in range(len(gen_imgs)):
                        save_image(gen_imgs[j]*0.5 + 0.5,
                                   "%s/%i_res%i.png" % (dest_dir, kwargs['epoch'], j),
                                   nrow=20)
                else:
                    save_image(gen_imgs*0.5 + 0.5,
                               "%s/%i.png" % (dest_dir, kwargs['epoch']),
                               nrow=20)
        return {}
    return _dump_img_handler
