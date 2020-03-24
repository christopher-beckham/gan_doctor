## GAN doctor

- Easy to implement new GAN classes with minimal effort and run them
- Out-of-the-box support for IS/FID metrics (both PyTorch and TensorFlow) which are periodically monitored during training rather than post-hoc
- Strong baseline (in this case spectral normed JSGAN, courtesy of X) implemented which achieves inception score of ...

### How to run

```
#!/bin/bash

cd ..

export RESULTS_DIR=/results/gan_doctor/

python task_launcher.py \
--gen=networks/cosgrove/gen_old.py \
--disc=networks/cosgrove/disc_old.py \
--disc_args="{'spec_norm': True, 'sigmoid': False, 'nf': 32}" \
--gen_args="{'nf': 32}" \
--gan=models/gan.py \
--gan_args="{'loss': 'hinge'}" \
--trial_id=999999 \
--name=cifar10_simple_hinge \
--val_batch_size=32 \
--z_dim=256 \
--img_size=32 \
--dataset=iterators/cifar10.py \
--compute_is_every=1 \
--n_samples_is=50000 \
--use_tf_metrics
```

Experiments are structured like so: $RESULTS_DIR/<name>/<trial_id>, where <name> is the `--name` argument and 
trial_id is determined by SLURM_JOB_ID (if not manually defined by `--trial_id`).
  
Run `task_launcher.py --help` for detailed descriptions of each argument. Perhaps what is most useful to know is that the `--gan` argument defines a GAN class. If you look at the one used above (`--gan=models/gan.py`) you'll see that it defines a `train_on_instance(self, z, y)` method, which gets called by its base class on each minibatch, and performs a gradient step. This makes it easy to define new GAN classes: you simply write a new one with your own `train_on_instance` method.

### Analysing results

In the experiments directory, `results.json` is txt file consisting of lines of json strings, one per epoch. The json string is a dictionary of various metrics that have been logged during training (you can easily define your own). You can convert this to a pandas data frame by running this code:

```
TODO

plt.plot(df['train_is_mean']) # inception score
plt.plot(df['train_fid_mean']) # fid score
```



Notes:
- FID / Inception numbers from TensorFlow implementations cannot be compared to PyTorch. (See consistency GAN paper) Using the flag `--use_tf_metrics` will compute the TensorFlow IS/FID metrics but this may be slower to invoke than the corresponding PyTorch ones. Also, because it takes a while to compute you should only do this every X epochs, which can be set with the flag `--compute_is_every=X`.
- Each GAN class has a `train_on_instance(self, x, z)` method, this gets called per minibatch and this is where you perform your forward and backward passes. See models/gan.py for how the baseline is implemented.
- Pay attention to the hyperparameters beta1,beta2 of ADAM, these can make a big difference and different GAN implementations use different ones.
- Do not remove spectral norm from the discriminator unless you know what you're doing.
