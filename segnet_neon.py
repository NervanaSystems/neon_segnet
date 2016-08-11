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
from mask_iterator import MaskIterator
global _get_bprop_upsampling


def gen_model(num_channels, height, width):
    assert NervanaObject.be is not None, 'need to generate a backend before using this function'

    init_uni = Kaiming()

    # we have 1 issue, they have bias layers we don't allow batchnorm and biases
    conv_common = dict(padding=1, init=init_uni, activation=Rectlin(), batch_norm=True)

    # set up the layers
    layers = []

    # need to store a ref to the pooling layers to pass
    # to the upsampling layers to get the argmax indicies
    # for upsampling, this stack holds the pooling layer refs
    pool_layers = []

    # first loop generates the encoder layers
    nchan = [64, 128, 256, 512, 512]
    for ind in range(len(nchan)):
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
    for ind in range(len(nchan)-1,-1,-1):
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
    act_last = PixelwiseSoftmax(num_channels, height, width, name="PixelwiseSoftmax")
    conv_last = dict(padding=1, init=init_uni, activation=act_last, batch_norm=False)
    layers.append(Conv((3, 3, num_channels), strides=1, name='deconv_out', **conv_last))
    return layers


def main():
    # larger batch sizes may not fit on GPU
    parser = NeonArgparser(__doc__, default_overrides={'batch_size': 4})

    args = parser.parse_args(gen_be=False)

    (c, h, w) = (2, 256, 256)

    # need to use the backend with the new upsampling layer implementation
    be = NervanaGPU_Upsample(rng_seed=args.rng_seed,
                             device_id=args.device_id)
    # set batch size
    be.bsz = args.batch_size
    args.data_dir = '/usr/local/data/I1K/macrobatches/'
    args.num_classes = 2
    args.height = 256
    args.width = 256

    # couple backend to global neon object
    NervanaObject.be = be

    shape = dict(channel_count=3, height=h, width=w, subtract_mean=False)
    train_params = ImageParams(center=True, flip=False,
                               scale_min=min(h, w), scale_max=min(h, w),
                               aspect_ratio=0, **shape)
    test_params = ImageParams(center=True, flip=False,
                              scale_min=min(h, w), scale_max=min(h, w),
                              aspect_ratio=0, **shape)
    common = dict(target_size=h*w, target_conversion='read_contents',
                  onehot=False, target_dtype=np.uint8, nclasses=c)

    img_set_options = dict(repo_dir='/usr/local/data/I1K/macrobatches/',
                           inner_size=args.height,
                           subset_pct=20)

    train_set = MaskIterator(10, 25, 2, set_name='train',
                             scale_range=(256, 256), **img_set_options)
    val_set = MaskIterator(10, 25, 2, set_name='validation',
                           scale_range=(256, 256), do_transforms=False,
                           **img_set_options)

    # initialize model object
    layers = gen_model(c, h, w)
    segnet_model = Model(layers=layers)
    cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    segnet_model.initialize(train_set, cost=cost)
    print segnet_model

    from neon.util.persist import load_obj
    from neon.util.modeldesc import ModelDescription
    print 'loading weights...'
    model_dict = load_obj('/home/users/evren/kaggle/driving/localization/VGG_D.p')
    #model_dict = load_obj(args.model_file)
    model_desc = ModelDescription(model_dict)

    from copy import deepcopy
    for ll in segnet_model.layers_to_optimize:
        trained_layer = model_desc.getlayer(ll.name)
        if trained_layer is not None:
            print ll.name
            layer_config = deepcopy(trained_layer)
            ll.load_weights(layer_config)

    # configure callbacks
    callbacks = Callbacks(segnet_model, eval_set=val_set, **args.callback_args)

    opt_gdm = GradientDescentMomentum(1.0e-6, 0.9, wdecay=0.0005, schedule=Schedule())
    opt_biases = GradientDescentMomentum(2.0e-6, 0.9, schedule=Schedule())
    opt_bn = GradientDescentMomentum(1.0e-6, 0.9, schedule=Schedule())
    opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases, 'BatchNorm': opt_bn})


    segnet_model.fit(train_set, optimizer=opt, num_epochs=args.epochs,
                     cost=cost, callbacks=callbacks)

    # get the trained segnet model outputs for valisation set
    outs_val = segnet_model.get_outputs(val_set)

    with open('outputs.pkl', 'w') as fid:
        pickle.dump(outs_val, fid, -1)

if __name__ == '__main__':
    main()
