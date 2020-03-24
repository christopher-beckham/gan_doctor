import sys
import numpy as np
import torch
import os
import argparse
import glob
import yaml
import random
import string
from skimage.io import imsave
from importlib import import_module
from torch.utils.data import (DataLoader,
                              Subset)
from utils import (generate_name_from_args,
                   find_latest_pkl_in_folder,
                   count_params)
from handlers import (fid_handler,
                      is_handler,
                      dump_img_handler)
from torchvision.utils import save_image

use_shuriken = False
try:
    from shuriken.utils import get_hparams
    use_shuriken = True
except:
    print("WARNING: tried to import Shuriken but failed")


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
        parser.add_argument('--img_size', type=int, default=32)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--z_dim', type=int, default=62)
        parser.add_argument('--lr_g', type=float, default=2e-4)
        parser.add_argument('--lr_d', type=float, default=2e-4)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--beta2', type=float, default=0.999)
        parser.add_argument('--update_g_every', type=int, default=1)
        parser.add_argument('--dataset', type=str, default="iterators/cifar10.py")
        parser.add_argument('--resume', type=str, default='auto')

        parser.add_argument('--gen', type=str, default="networks/cosgrove/gen.py")
        parser.add_argument('--disc', type=str, default="networks/cosgrove/disc.py")
        parser.add_argument('--gen_args', type=str, default=None)
        parser.add_argument('--disc_args', type=str, default=None)
        parser.add_argument('--subset_train', type=int, default=None)

        parser.add_argument('--save_path', type=str, default=None)
        parser.add_argument('--save_images_every', type=int, default=100)
        parser.add_argument('--save_every', type=int, default=10)
        parser.add_argument('--val_batch_size', type=int, default=512)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--compute_is_every', type=int, default=1)
        parser.add_argument('--n_samples_is', type=int, default=5000)
        parser.add_argument('--use_tf_metrics', action='store_true')
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--mode', type=str, choices=['train',
                                                         'eval_is_tf',
                                                         'eval_fid_tf'],
                            default='train')
        args = parser.parse_args()
        return args

    args = parse_args()
    args = vars(args)

    if use_shuriken:
        shk_args = get_hparams()
        print("Args passed from Shk:", shk_args)
        for key in shk_args:
            #if key not in KWARGS_FOR_NAME:
            #    raise Exception("Shuriken-supplied HP %s is unknown" % key)
            if shk_args[key] == '':
                shk_args[key] = True
        args.update(shk_args)

    if args['trial_id'] is None and 'SHK_TRIAL_ID' in os.environ:
        print("SHK_TRIAL_ID found so injecting this into `trial_id`...")
        args['trial_id'] = os.environ['SHK_TRIAL_ID']

    if args['name'] is None and 'SHK_EXPERIMENT_ID' in os.environ:
        print("SHK_EXPERIMENT_ID found so injecting this into `name`...")
        args['name'] = os.environ['SHK_EXPERIMENT_ID']

    if 'EAI_SEED' in os.environ:
        print("EAI_SEED found so overriding args.seed...")
        args['seed'] = int(os.environ['EAI_SEED'])

    print("** ARGUMENTS **")
    print("  " + yaml.dump(args).replace("\n", "\n  "))

    if args['save_path'] is None:
        args['save_path'] = os.environ['RESULTS_DIR']

    # <save_path>/<seed>/<name>/_trial=<trial>,...,...,
    if args['name'] is None:
        save_path = "%s/s%i" % (args['save_path'], args['seed'])
    else:
        save_path = "%s/s%i/%s" % (args['save_path'], args['seed'], args['name'])

    name = args['trial_id']
    print("Auto-generated experiment name: %s" % name)
    print("Save path: %s" % save_path)

    expt_path = "%s/%s" % (save_path, name)
    print("Full path: %s" % expt_path)

    if args['mode'] == 'train':
        torch.manual_seed(args['seed'])

    gen_args = eval(args['gen_args']) if 'gen_args' in args else {}
    disc_args = eval(args['disc_args']) if 'disc_args' in args else {}

    module_gen = import_module(args['gen'].replace("/", ".").\
                               replace(".py", ""))
    gen_fn = module_gen.get_network(z_dim=args['z_dim'],
                                    **eval(args['gen_args']))

    module_disc = import_module(args['disc'].replace("/", ".").\
                                replace(".py", ""))
    disc_fn = module_disc.get_network(**eval(args['disc_args']))

    print("Generator:")
    print(gen_fn)
    print("(%i params)" % count_params(gen_fn))
    print("Discriminator:")
    print(disc_fn)
    print("(%i params)" % count_params(disc_fn))

    module_gan = import_module(args['gan'].replace("/", ".").\
                               replace(".py", ""))
    gan_class = module_gan.get_class()

    module_dataset = import_module(args['dataset'].replace("/", ".").\
                                   replace(".py", ""))

    # Load training set.
    dataset_train = module_dataset.get_dataset_train(img_size=args['img_size'])
    # Load test set (for IS/FID calculation)
    dataset_test = module_dataset.get_dataset_test(img_size=args['img_size'])

    if args['subset_train'] is not None:
        # The subset is randomly sampled from the
        # training data, and changes depending on
        # the seed.
        indices = np.arange(0, args['subset_train'])
        rs = np.random.RandomState(args['seed'])
        rs.shuffle(indices)
        indices = indices[0:args['subset_train']]
        dataset_train = Subset(dataset_train, indices=indices)

    loader_train = DataLoader(dataset_train,
                              shuffle=True,
                              batch_size=args['batch_size'],
                              num_workers=args['num_workers'])
    loader_test = DataLoader(dataset_test,
                             shuffle=False,
                             batch_size=args['val_batch_size'],
                             num_workers=0)

    # TODO: support different optim flags
    # for opt_g and opt_d
    handlers = []
    gan_kwargs = {
        'gen_fn': gen_fn,
        'disc_fn': disc_fn,
        'z_dim': args['z_dim'],
        'update_g_every': args['update_g_every'],
        'opt_d_args': {'lr': args['lr_d'],
                       'betas': (args['beta1'], args['beta2'])},
        'opt_g_args': {'lr': args['lr_g'],
                       'betas': (args['beta1'], args['beta2'])},
        'handlers': handlers
    }

    gan_kwargs_from_args = {}
    if args['gan_args'] is not None:
        gan_kwargs_from_args = eval(args['gan_args'])
    gan_kwargs.update(gan_kwargs_from_args)
    gan = gan_class(**gan_kwargs)

    print("gen optim")
    print(gan.optim['g'])

    print("d optim")
    print(gan.optim['d'])

    if args['compute_is_every'] > 0:

        use_tf = args['use_tf_metrics']

        print("IS/FID handler: bs=%i, n_samples=%i" % \
              (args['val_batch_size'], args['n_samples_is']))

        handlers.append(
            is_handler(gan,
                       use_tf=use_tf,
                       batch_size=args['val_batch_size'],
                       n_samples=args['n_samples_is'],
                       eval_every=args['compute_is_every'])
        )

        handlers.append(
            fid_handler(gan,
                        cls=None,
                        use_tf=use_tf,
                        batch_size=args['val_batch_size'],
                        n_samples=args['n_samples_is'],
                        loader=loader_test,
                        eval_every=args['compute_is_every'])
        )

    handlers.append(
        dump_img_handler(gan,
                         batch_size=args['val_batch_size'],
                         dest_dir="%s/%s" % (save_path, name))
    )

    print("List of handlers:")
    print(handlers)

    if not os.path.exists(expt_path):
        os.makedirs(expt_path)

    with open("%s/cfg.yaml" % expt_path, "w") as f:
        f.write(yaml.dump(args))

    # Dump some test images.
    x, _ = iter(loader_train).next()
    save_image(x*0.5 + 0.5, "%s/real_samples.png" % (expt_path))

    if args['resume'] is not None:
        if args['resume'] == 'auto':
            # autoresume
            latest_model = find_latest_pkl_in_folder(expt_path)
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
            itr=loader_train,
            epochs=args['epochs'],
            model_dir=expt_path,
            result_dir=expt_path,
            save_every=args['save_every']
        )
    elif args['mode'] == 'eval_is_tf':

        from utils import compute_is_tf

        gan._eval()

        test_samples = []
        for b, (x_batch, _) in enumerate(loader_test):
            test_samples.append(x_batch.numpy())
        test_samples = np.vstack(test_samples)

        dd = compute_is_tf(500,
                           gan,
                           args['val_batch_size'])

    elif args['mode'] == 'eval_fid_tf':

        from utils import compute_fid_tf

        gan._eval()

        # NOTE: this one actually uses precomputed
        # FIDs (on train/test/both?) internally
        # in an npz file, so no need to extract
        # test examples here.
        #test_samples = []
        #for b, (x_batch, _) in enumerate(loader_test):
        #    test_samples.append(x_batch.numpy())
        #test_samples = np.vstack(test_samples)

        dd = compute_fid_tf(n_gan_samples=500,
                            gan=gan,
                            batch_size=args['val_batch_size'])

        print(dd)

    else:
        raise Exception("")
