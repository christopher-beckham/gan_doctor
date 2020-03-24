import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from collections import OrderedDict
import glob

from fid_score import calculate_fid_given_imgs



def find_latest_pkl_in_folder(model_dir):
    # List all the pkl files.
    files = glob.glob("%s/*.pkl" % model_dir)
    # Make them absolute paths.
    files = [os.path.abspath(key) for key in files]
    if len(files) > 0:
        # Get creation time and use that.
        latest_model = max(files, key=os.path.getctime)
        print("Auto-resume mode found latest model: %s" %
              latest_model)
        return latest_model
    return None

def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def is_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False

def count_params(module, trainable_only=True):
    """Count the number of parameters in a
    module.

    :param module: PyTorch module
    :param trainable_only: only count trainable
      parameters.
    :returns: number of parameters
    :rtype:

    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num

def generate_name_from_args(dd, kwargs_for_name):
    buf = {}
    for key in dd:
        if key in kwargs_for_name:
            if dd[key] is None:
                continue
            new_name, fn_to_apply = kwargs_for_name[key]
            new_val = fn_to_apply(dd[key])
            if dd[key] is True:
                new_val = ''
            buf[new_name] = new_val
    buf_sorted = OrderedDict(sorted(buf.items()))
    #tags = sorted(tags.split(","))
    name = ",".join([ "%s=%s" % (key, buf_sorted[key]) for key in buf_sorted.keys()])
    return name

def bce_loss(prediction, target):
    if not hasattr(target, '__len__'):
        target = torch.ones_like(prediction)*target
        if prediction.is_cuda:
            target = target.cuda()
    loss = torch.nn.BCELoss()
    if prediction.is_cuda:
        loss = loss.cuda()
    return loss(prediction, target)

def hinge_loss_d(real, fake):
    return torch.mean(torch.nn.ReLU()(1. - real)) + \
        torch.mean(torch.nn.ReLU()(1. + fake))

def hinge_loss_g(fake):
    return -torch.mean(fake)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def dump_samples_to_disk(folder, how_many, bs):
    if not os.path.exists(folder):
        os.makedirs(folder)
    train_size = how_many
    for b in range( (train_size // bs) + 1 ):
        print("Generating fake samples -- iteration %i" % b)
        samples_ = net.sample(bs).data.cpu().numpy()*0.5 + 0.5
        samples_ = (samples_*255.).astype("uint8")
        if b == 0:
            samples = samples_
        else:
            samples = np.vstack((samples, samples_))
    samples = samples[0:train_size]
    assert len(samples) == train_size
    for i in range(samples.shape[0]):
        img = samples[i]
        imsave(arr=img.swapaxes(0,1).swapaxes(1,2),
               fname="%s/%i.png" % (folder,i))

def _extract_loader(loader):
    '''Extract images from data loader and de-norm them
    and convert into int32.'''
    train_samples = []
    for x_batch, _ in loader:
        train_samples.append(x_batch)
    train_samples = np.vstack(train_samples)
    return train_samples

def _extract_samples(gan, batch_size, n_samples, verbose=False):

    print("bs==", batch_size)
    #import pdb
    #pdb.set_trace()

    '''Extract images from the GAN, and de-norm them
    and convert into int32.
    '''
    n_batches = int(np.ceil(n_samples / batch_size))
    gen_samples = []
    for _ in range(n_batches):
        if verbose:
            print("iter: ", (_+1), "/", n_batches)
        with torch.no_grad():
            samples = gan.sample(batch_size)
        if type(samples) == list:
            samples = samples[-1]
        gen_samples.append(samples.cpu())
    gen_samples = torch.cat(gen_samples, dim=0)[0:n_samples].numpy()
    #gen_samples = gen_samples*0.5 + 0.5
    #gen_samples = (((gen_samples*0.5) + 0.5)*255.).astype(np.int32)
    return gen_samples

def compute_fid(train_samples,
                n_gan_samples,
                gan,
                batch_size,
                cls,
                verbose=False):

    use_cuda = gan.use_cuda

    gen_samples = _extract_samples(gan,
                                   batch_size,
                                   n_samples=n_gan_samples)

    assert len(gen_samples) == n_gan_samples

    # The method in fid_score.py expects the images to be in
    # [0,1], so scale it here.
    gen_samples = gen_samples*0.5 + 0.5

    print("n samples from test:", len(train_samples))
    print("n samples from GAN:", len(gen_samples))

    # train_samples is already in [0,1]
    #train_samples = train_samples*0.5 + 0.5

    score = calculate_fid_given_imgs(imgs1=train_samples,
                                     imgs2=gen_samples,
                                     batch_size=batch_size,
                                     cuda=use_cuda,
                                     model=cls)
    return {
        'fid': np.mean(score)
    }

def compute_fid_tf(n_gan_samples,
                   gan,
                   batch_size,
                   verbose=False):

    use_cuda = gan.use_cuda

    import tf.fid as tf_fid
    import tensorflow as tf

    print("inception fid score...>>>>>>")

    gen_samples = _extract_samples(gan,
                                   batch_size,
                                   n_samples=n_gan_samples)

    assert len(gen_samples) == n_gan_samples

    samples_tf = (((gen_samples*0.5)+0.5)*255.).astype(np.uint8)
    samples_tf = samples_tf.swapaxes(1, 2).swapaxes(2, 3)

    g = tf.Graph()
    with g.as_default():

        score = tf_fid.calculate_fid_given_imgs(
            samples_tf,
            "tf/cifar-10-fid.npz",
            batch_size=batch_size
        )

    return {
        'fid_tf': np.mean(score).astype(np.float64)
    }



def compute_is(n_samples,
               gan,
               batch_size):

    import inception_score

    gen_samples = _extract_samples(gan,
                                   batch_size,
                                   n_samples=n_samples)

    # The method in inception_score.py expects the images to be
    # in [-1, 1], so no need to scale here.

    print("foo")

    score_mu, score_std = inception_score.inception_score(
        imgs=gen_samples,
        batch_size=batch_size,
        resize=True,
        splits=10,
        pbar=True
    )
    #print("PyTorch Inception score: %f +/- %f" % (score_mu, score_std))
    return {
        'is_mean': np.mean(score_mu),
        'is_std': np.std(score_std)
    }

def compute_is_tf(n_samples,
                  gan,
                  batch_size):

    import tf.inception as tf_inception
    import tensorflow as tf

    print("inception tf score...>>>>>>")

    gen_samples = _extract_samples(gan,
                                   batch_size,
                                   n_samples=n_samples)

    assert len(gen_samples) == n_samples

    # The method in fid_score.py expects the images to be in
    # [0,1], so scale it here.
    gen_samples = gen_samples*0.5 + 0.5

    print("n samples from GAN:", len(gen_samples))

    # This method expects samples in [0,255]
    samples_tf = (((gen_samples*0.5)+0.5)*255.).astype(np.uint8)
    samples_tf = samples_tf.swapaxes(1, 2).swapaxes(2, 3)
    samples_tf = [ samples_tf[i] for i in range(len(samples_tf)) ]

    g = tf.Graph()
    with g.as_default():
        inception_score_mean, inception_score_std = \
            tf_inception.get_inception_score(samples_tf,
                                             batch_size=batch_size)

    return {
        'is_mean_tf': np.mean(inception_score_mean).astype(np.float64),
        'is_std_tf': np.mean(inception_score_std).astype(np.float64)
    }
