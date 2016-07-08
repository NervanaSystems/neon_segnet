# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import os
import sys
import h5py
import pickle
import logging
import numpy as np

from neon import NervanaObject
from neon.callbacks.callbacks import Callbacks
from neon.data import ImageParams
from neon.initializers import  Kaiming
from neon.layers import Conv, GeneralizedCost, Dropout, Pooling
from neon.models import Model
from neon.optimizers import MultiOptimizer, Schedule, GradientDescentMomentum
from neon.transforms import Rectlin, CrossEntropyMulti
from neon.util.argparser import NeonArgparser, extract_valid_args
from pixelwise_sm import PixelwiseSoftmax

from segnet_neon_backend import NervanaGPU_Upsample
from upsampling_layer import Upsampling
from segnet_neon_backend import _get_bprop_upsampling
from pixelwise_dataloader import PixelWiseImageLoader
global _get_bprop_upsampling


def gen_model():
    assert NervanaObject.be is not None, 'need to generate a backend before using this function'

    init_uni = Kaiming()

    # we have 1 issue, they have bias layers we don't allow batchnorm and biases
    conv_common = dict(padding=1, init=init_uni, activation=Rectlin(), batch_norm=True)

    # set up the layers
    layers=[]

    # need to store a ref to the pooling layers to pass
    # to the upsampling layers to get the argmax indicies
    # for upsampling, this stack holds the pooling layer refs
    pool_layers = []

    # first loop generates the encoder layers
    nchan = [64, 128, 256]
    nchan.extend([512]*2)
    for ind in range(5):
        nchanu = nchan[ind]
        lrng = 2 if ind <= 1 else 3
        for lind in range(lrng):
            nm = 'conv%d_%d' % (ind+1, lind+1)
            layers.append(Conv((3, 3, nchanu), strides=1, name=nm, **conv_common))

        layers.append(Pooling(2, strides=2, name='conv%d_pool' % ind))
        pool_layers.append(layers[-1])
        if ind >= 2:
            layers.append(Dropout(keep=0.5, name='drop%d' % (ind+1)))

    # this loop generates the decoder layers
    for ind in range(4,-1,-1):
        nchanu = nchan[ind]
        lrng = 2 if ind <= 1 else 3
        # upsampling layers need a ref to the corresponding pooling layer
        # to access the argmx indices for upsampling
        layers.append(Upsampling(2, pool_layers.pop(), strides=2, padding=0,
                      name='conv%d_unpool' % ind))
        for lind in range(lrng):
            nm = 'deconv%d_%d' % (ind+1, lind+1)
            if ind < 4 and lind == lrng-1:
                nchanu = nchan[ind]/2
            layers.append(Conv((3, 3, nchanu), strides=1, name=nm, **conv_common))
            if ind == 0:
                break
        if ind >= 2:
            layers.append(Dropout(keep=0.5, name='drop%d' % (ind+1)))

    # last conv layer outputs 12 channels, 1 for each output class
    # with a pixelwise softmax over the channels
    c, h, w = (12, 256, 512)
    act_last = PixelwiseSoftmax(c, h, w, name="PixelwiseSoftmax")
    conv_last = dict(padding=1, init=init_uni, activation=act_last, batch_norm=False)
    layers.append(Conv((3, 3, c), strides=1, name='deconv_out', **conv_last))
    return layers


def main():
    # larger batch sizes may not fit on GPU
    parser = NeonArgparser(__doc__, default_overrides={'batch_size': 4})
    parser.add_argument("--bench", action="store_true", help="run benchmark instead of training")
    args = parser.parse_args(gen_be=False)

    # need to use the backend with the new upsampling layer implementation
    be = NervanaGPU_Upsample(rng_seed=args.rng_seed,
                             device_id=args.device_id)
    # set batch size
    be.bsz = args.batch_size

    # couple backend to global neon object
    NervanaObject.be = be

    shape = dict(channel_count=3, height=256, width=512, subtract_mean=False)
    train_params = ImageParams(center=True, flip=False,
                               scale_min=256, scale_max=256,
                               aspect_ratio=0, **shape)
    test_params = ImageParams(center=True, flip=False,
                              scale_min=256, scale_max=256,
                              aspect_ratio=0, **shape)
    common = dict(target_size=256*512, target_conversion='read_contents',
                  onehot=False, target_dtype=np.uint8, nclasses=12)

    train_set = PixelWiseImageLoader(set_name='train', repo_dir=args.data_dir,
                                      media_params=train_params,
                                      shuffle=False, subset_percent=100,
                                      index_file=os.path.join(args.data_dir, 'train_images.csv'),
                                      **common)
    val_set = PixelWiseImageLoader(set_name='val', repo_dir=args.data_dir,media_params=test_params, 
                      index_file=os.path.join(args.data_dir, 'val_images.csv'), **common)

    # initialize model object
    layers = gen_model()
    segnet_model = Model(layers=layers)

    # configure callbacks
    callbacks = Callbacks(segnet_model, eval_set=val_set, **args.callback_args)

    opt_gdm = GradientDescentMomentum(1.0e-6, 0.9, wdecay=0.0005, schedule=Schedule())
    opt_biases = GradientDescentMomentum(2.0e-6, 0.9, schedule=Schedule())
    opt_bn = GradientDescentMomentum(1.0e-6, 0.9, schedule=Schedule())
    opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases, 'BatchNorm': opt_bn})

    cost = GeneralizedCost(costfunc=CrossEntropyMulti())

    if args.bench:
        segnet_model.initialize(train_set, cost=cost)
        segnet_model.benchmark(train_set, cost=cost, optimizer=opt)
        sys.exit(0)
    else:
        segnet_model.fit(train_set, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

    # get the trained segnet model outputs for valisation set
    outs_val = segnet_model.get_outputs(val_set)

    with open('outputs.pkl', 'w') as fid:
        pickle.dump(outs_val, fid, -1)

if __name__ == '__main__':
    main()
