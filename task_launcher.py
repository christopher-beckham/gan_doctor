import sys
import numpy as np
import torch
import os
import argparse
import glob
import yaml
from skimage.io import imsave
from importlib import import_module
from torch.utils.data import DataLoader
from utils import (generate_name_from_args,
                   find_latest_pkl_in_folder)
from handlers import (fid_handler,
                      is_handler,
                      dump_img_handler)

use_shuriken = False
try:
    from shuriken.utils import get_hparams
    use_shuriken = True
except:
    pass

# This dictionary's keys are the ones that are used
# to auto-generate the experiment name. The values
# of those keys are tuples, the first element being
# shortened version of the key (e.g. 'dataset' -> 'ds')
# and a function which may optionally shorten the value.
id_ = lambda x: str(x)
dict2line = lambda x: x.replace(" ", "").\
    replace('"', '').\
    replace("'", "").\
    replace(":", "=").\
    replace(",", ";")
KWARGS_FOR_NAME = {
    'gan': ('gan', lambda x: os.path.basename(x)),
    'gan_args': ('gan_args', dict2line),
    'dataset': ('ds', lambda x: os.path.basename(x)),
    'network': ('net', lambda x: os.path.basename(x)),
    'batch_size': ('bs', id_),
    'n_channels': ('nc', id_),
    'ngf': ('ngf', id_),
    'ndf': ('ndf', id_),
    'lr': ('lr', id_),
    'update_g_every': ('g', id_),
    'beta1': ('b1', id_),
    'beta2': ('b2', id_),
    'trial_id': ('_trial', id_)
}

if __name__ == '__main__':

    '''
    Process arguments.
    '''
    def parse_args():
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--name', type=str, default=None)
        parser.add_argument('--trial_id', type=str, default=None)
        parser.add_argument('--gan', type=str, default="models/gan.py")
        parser.add_argument('--gan_args', type=str, default=None)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--loss', type=str, default='jsgan')
        parser.add_argument('--z_dim', type=int, default=62)
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--beta2', type=float, default=0.999)
        parser.add_argument('--update_g_every', type=int, default=1)
        parser.add_argument('--dataset', type=str, default="iterators/cifar10.py")
        parser.add_argument('--resume', type=str, default='auto')
        parser.add_argument('--network', type=str, default="networks/mnist.py")
        parser.add_argument('--save_path', type=str, default='./results')
        parser.add_argument('--save_images_every', type=int, default=100)
        parser.add_argument('--save_every', type=int, default=10)
        parser.add_argument('--val_batch_size', type=int, default=512)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--mode', type=str, choices=['train', 'pdb'],
                            default='train')
        args = parser.parse_args()
        return args

    args = parse_args()
    args = vars(args)

    print("args before shuriken:")
    print(args)

    if use_shuriken:
        shk_args = get_hparams()
        print("shk args:", shk_args)
        # Stupid bug that I have to fix: if an arg is ''
        # then assume it's a boolean.
        for key in shk_args:
            if shk_args[key] == '':
                shk_args[key] = True
        args.update(shk_args)

    if args['trial_id'] is None and 'SHK_TRIAL_ID' in os.environ:
        print("SHK_TRIAL_ID found so injecting this into `trial_id`...")
        args['trial_id'] = os.environ['SHK_TRIAL_ID']

    if args['name'] is None and 'SHK_EXPERIMENT_ID' in os.environ:
        print("SHK_EXPERIMENT_ID found so injecting this into `name`...")
        args['name'] = os.environ['SHK_EXPERIMENT_ID']

    print("** ARGUMENTS **")
    print("  " + yaml.dump(args).replace("\n", "\n  "))

    if args['save_path'] is None:
        args['save_path'] = os.environ['RESULTS_DIR']

    # <save_path>/<seed>/<name>/_trial=<trial>,...,...,
    if args['name'] is None:
        save_path = "%s/s%i" % (args['save_path'], args['seed'])
    else:
        save_path = "%s/s%i/%s" % (args['save_path'], args['seed'], args['name'])

    # TODO: also add gan class specific args to this too...
    name = generate_name_from_args(args, KWARGS_FOR_NAME)
    print("Auto-generated experiment name: %s" % name)
    print("Save path: %s" % save_path)
    print("Full path: %s/%s" % (save_path, name))

    if args['mode'] == 'train':
        torch.manual_seed(args['seed'])

    module_net = import_module(args['network'].replace("/", ".").\
                               replace(".py", ""))
    gen_fn, disc_fn = module_net.get_network(args['z_dim'])

    module_gan = import_module(args['gan'].replace("/", ".").\
                               replace(".py", ""))
    gan_class = module_gan.GAN

    module_dataset = import_module(args['dataset'].replace("/", ".").\
                                   replace(".py", ""))
    dataset = module_dataset.get_dataset()
    loader = DataLoader(dataset,
                        shuffle=True,
                        batch_size=args['batch_size'])


    # TODO: support different optim flags
    # for opt_g and opt_d
    handlers = []
    gan_kwargs = {
        'gen_fn': gen_fn,
        'disc_fn': disc_fn,
        'z_dim': args['z_dim'],
        'update_g_every': args['update_g_every'],
        'opt_d_args': {'lr': args['lr'],
                       'betas': (args['beta1'], args['beta2'])},
        'opt_g_args': {'lr': args['lr'],
                       'betas': (args['beta1'], args['beta2'])},
        'handlers': handlers
    }
    if args['gan_args'] is not None:
        gan_kwargs_from_args = eval(args['gan_args'])
        gan_kwargs.update(gan_kwargs_from_args)
    gan = gan_class(**gan_kwargs)

    loader_handler = DataLoader(
        dataset,
        shuffle=True,
        batch_size=args['val_batch_size']
    )
    handlers.append(
        is_handler(gan,
                   batch_size=args['val_batch_size'],
                   loader=loader_handler)
    )

    handlers.append(
        fid_handler(gan,
                    cls=None,
                    batch_size=args['val_batch_size'],
                    loader=loader_handler)
    )

    handlers.append(
        dump_img_handler(gan,
                         batch_size=args['val_batch_size'],
                         dest_dir="%s/%s" % (save_path, name))
    )

    print("List of handlers:")
    print(handlers)

    if args['resume'] is not None:
        if args['resume'] == 'auto':
            # autoresume
            model_dir = "%s/%s" % (args['save_path'], args['name'])
            latest_model = find_latest_pkl_in_folder(model_dir)
            if latest_model is not None:
                gan.load(latest_model)
        else:
            gan.load(args['resume'])

    """
    if args.interactive is not None:
        how_many = 5000*10
        if 'inception' in args.interactive:
            if args.interactive == 'inception':
                # Compute the Inception score and output
                # mean and std.
                compute_inception_cifar10(how_many=how_many)
            elif args.interactive == 'inception_tf':
                compute_inception_cifar10(how_many=how_many, use_tf=True)
            elif args.interactive == 'inception_both':
                compute_inception_cifar10(how_many=how_many)
                compute_inception_cifar10(how_many=how_many, use_tf=True)
        elif args.interactive == 'dump':
            # Dump the images to disk.
            compute_inception_cifar10(how_many=how_many,
                                      seed=0,
                                      dump_only=True)
        elif args.interactive == 'dump_to_disk':
            dump_samples_to_disk(folder="img_dump", how_many=50000, bs=512)
        elif args.interactive == 'fid_tf':
            compute_fid_cifar10(use_tf=True)
        elif args.interactive == 'free':
            import pdb
            pdb.set_trace()
    else:
    """

    if args['mode'] == 'train':
        gan.train(
            itr=loader,
            epochs=args['epochs'],
            model_dir="%s/%s" % (save_path, name),
            result_dir="%s/%s" % (save_path, name),
            save_every=args['save_every']
        )
