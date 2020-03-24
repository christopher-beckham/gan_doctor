#!/bin/bash

cd ..

export RESULTS_DIR=/results/gan_doctor/

#todo
export SLURM_TMP_DIR=/tmp

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/:/usr/local/cuda-10.1/targets/x86_64-linux/lib/:${LD_LIBRARY_PATH}"

CUDA_LAUNCH_BLOCKING=1 python task_launcher.py \
--gen=networks/cosgrove/gen_old.py \
--disc=networks/cosgrove/disc_old.py \
--disc_args="{'spec_norm': True, 'sigmoid': False, 'nf': 32}" \
--gen_args="{'nf': 32}" \
--gan=models/rgan.py \
--gan_args="{'loss': 'hinge'}" \
--trial_id=999999 \
--name=cifar10_simple_hinge_rgan \
--val_batch_size=32 \
--z_dim=256 \
--img_size=32 \
--dataset=iterators/cifar10.py \
--compute_is_every=1 \
--n_samples_is=500 \
--use_tf_metrics \
--subset_train=128
