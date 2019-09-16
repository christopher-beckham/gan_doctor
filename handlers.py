import torch
import pickle
import os
import numpy as np
from torchvision.utils import save_image
from utils import (compute_fid, compute_is)

def is_handler(gan,
               loader,
               batch_size=None,
               eval_every=5):
    '''Callback for computing Inception Score.

    Params
    ------

    '''
    if batch_size is None:
        batch_size = loader.batch_size
    def _is_handler(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1 and \
           kwargs['mode'] == 'train' and \
           kwargs['epoch'] % eval_every == 0:
            gan._eval()
            with torch.no_grad():
                return compute_is(loader=loader,
                                  gan=gan,
                                  batch_size=batch_size)
        return {}
    return _is_handler

def fid_handler(gan,
                cls,
                loader,
                batch_size=None,
                eval_every=5):
    '''Callback for computing FID score.

    Params
    ------

    '''
    if batch_size is None:
        batch_size = loader.batch_size
    def _fid_handler(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1 and \
           kwargs['mode'] == 'train' and \
           kwargs['epoch'] % eval_every == 0:
            gan._eval()
            with torch.no_grad():
                return compute_fid(loader=loader,
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
                save_image(gen_imgs*0.5 + 0.5,
                           "%s/%i.png" % (dest_dir, kwargs['epoch']),
                           nrow=20)
        return {}
    return _dump_img_handler
