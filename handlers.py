import torch
import pickle
import os
import numpy as np
from torchvision.utils import save_image
from utils import compute_fid

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
