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
from neon.layers import Conv, GeneralizedCost, Dropout, Pooling, Activation, MergeSum
from neon.layers import SkipNode
from neon.models import Model
from neon.optimizers import MultiOptimizer, Schedule, GradientDescentMomentum, Adadelta
from neon.transforms import Rectlin, MeanSquared 
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.data import DataLoader
from depth_iterator import DepthImageLoader

from segnet_neon_backend import NervanaGPU_Upsample
from upsampling_layer import Upsampling
from segnet_neon_backend import _get_bprop_upsampling
global _get_bprop_upsampling


def simple_mb(d1, d2):
    init = Kaiming()
    path1 = [
             Conv((1,1,d1), strides=1, padding=0, batch_norm=True, activation=Rectlin(), init=init),
             Conv((3,3,d1), strides=1, padding=1, batch_norm=True, activation=Rectlin(), init=init),
             Conv((1,1,d2), strides=1, padding=0, batch_norm=True, init=init)
            ]
    return MergeSum([path1, SkipNode()])

def complex_mb(d1, d2, stride):
    init = Kaiming()
    path1 = [
             Conv((1,1,d1), strides=stride, padding=0, batch_norm=True, activation=Rectlin(), init=init),
             Conv((3,3,d1), strides=1,      padding=1, batch_norm=True, activation=Rectlin(), init=init),
             Conv((1,1,d2), strides=1,      padding=0, batch_norm=True, init=init)
            ]

    path2 = [Conv((1,1,d2), strides=stride, padding=0, batch_norm=True, init=init)]
    return MergeSum([path1, path2])


def gen_model():
    assert NervanaObject.be is not None, 'need to generate a backend before using this function'

    init= Kaiming()

    # we have 1 issue, they have bias layers we don't allow batchnorm and biases
    #conv_common = dict(padding=1, init=init_uni, activation=Rectlin(), batch_norm=True)

    # set up the layers
    layers = []

    layers.append(Conv((7, 7, 64), init=init, strides=2, padding=3,
                       activation=None, batch_norm=True, name='conv1'))
    layers.append(Pooling(3, strides=2, padding=1, name='pool1'))
    layers.append(Activation(transform=Rectlin(), name='relu1'))

    layers.append(complex_mb(64, 256, 1))
    layers.append(simple_mb(64, 256))

    layers.append(complex_mb(128, 512, 2))
    layers.append(simple_mb(128, 512))
    layers.append(simple_mb(128, 512))
    layers.append(simple_mb(128, 512))

    layers.append(complex_mb(256, 1024, 2))
    for ind in range(5):
        layers.append(simple_mb(256, 1024))

    layers.append(complex_mb(512, 2048, 2))
    layers.append(simple_mb(512, 2048))
    layers.append(simple_mb(512, 2048))

    layers.append(Conv((1,1,1024), strides=1, padding=0, batch_norm=True, init=init))

    pads = [2, 1, 2, 2, 2]
    for ind in range(1, 6):
        layers.append(Upsampling(2, None, strides=2, padding=0, name='unpool%d' % ind))
        if ind == 1:
            pad_ = {'pad_h': 2, 'pad_w':1}
        elif ind == 2:
            pad_ = {'pad_h': 1, 'pad_w':2}
        elif ind == 4:
            pad_ = {'pad_h': 2, 'pad_w':3}
        else:
            pad_ = 2
        layers.append(Conv((5, 5, 2**(11-ind)), batch_norm=False, padding=pad_, strides=1,
                           init=init, activation=Rectlin()))

    layers.append(Conv((3,3, 1), batch_norm=False, activation=Rectlin(),
                  padding={'pad_h':1, 'pad_w':1}, 
                  init=init))

    return layers


def main():
    # larger batch sizes may not fit on GPU
    parser = NeonArgparser(__doc__, default_overrides={'batch_size': 4})

    args = parser.parse_args(gen_be=False)
    args.data_dir = '/usr/local/data/evren/kitti/training'

    h = 304 #256 #  219
    w = 228 #1024 # 1232

    t_h = h #158
    t_w = w #126

    # need to use the backend with the new upsampling layer implementation
    be = NervanaGPU_Upsample(rng_seed=args.rng_seed,
                             device_id=args.device_id)
    # set batch size
    be.bsz = args.batch_size

    # couple backend to global neon object
    NervanaObject.be = be

    shape = dict(channel_count=3, height=h, width=w, subtract_mean=False)
    train_params = ImageParams(center=True, flip=False,
                               scale_min=min(h, w), scale_max=min(h, w),
                               aspect_ratio=0, **shape)
    common = dict(target_size=t_h*t_w, target_conversion='read_contents',
                  onehot=False, target_dtype=np.uint8)
    dd='/usr/local/data/evren/kitti/training'

    left_set_train = DataLoader(set_name='left_depth_train', repo_dir=dd,
                                media_params=train_params,
                                shuffle=False, subset_percent=100,
                                index_file=os.path.join(dd, 'images_left_train.csv'),
                                **common)

    right_set_train = DataLoader(set_name='right_depth_train', repo_dir=dd,
                                 media_params=train_params,
                                 shuffle=False, subset_percent=100,
                                 index_file=os.path.join(dd, 'images_right_train.csv'),
                                 **common)

    left_set_val = DataLoader(set_name='left_depth_val', repo_dir=dd,
                                media_params=train_params,
                                shuffle=False, subset_percent=100,
                                index_file=os.path.join(dd, 'images_left_val.csv'),
                                **common)

    right_set_val = DataLoader(set_name='right_depth_val', repo_dir=dd,
                                 media_params=train_params,
                                 shuffle=False, subset_percent=100,
                                 index_file=os.path.join(dd, 'images_right_val.csv'),
                                 **common)

    train_set = DepthImageLoader([left_set_train, right_set_train], right_set_train.shape)
    val_set = DepthImageLoader([left_set_val, right_set_val], right_set_val.shape)

    # initialize model object
    layers = gen_model()
    segnet_model = Model(layers=layers)

    # configure callbacks
    callbacks = Callbacks(segnet_model, eval_set=val_set, **args.callback_args)

    #opt_gdm = GradientDescentMomentum(1.0e-10, 0.9, wdecay=0.0005, schedule=Schedule())
    #opt_biases = GradientDescentMomentum(2.0e-10, 0.9, schedule=Schedule())
    #opt_bn = GradientDescentMomentum(1e-10, 0.9, schedule=Schedule())
    #opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases, 'BatchNorm': opt_bn})
    opt = Adadelta()

    cost = GeneralizedCost(costfunc=MeanSquared())

    #segnet_model.initialize(train_set)
    #print segnet_model
    #import ipdb; ipdb.set_trace()
    segnet_model.fit(train_set, optimizer=opt,
                     num_epochs=args.epochs, cost=cost, callbacks=callbacks)

    # get the trained segnet model outputs for valisation set
    outs_val = segnet_model.get_outputs(val_set)

    with open('outputs.pkl', 'w') as fid:
        pickle.dump(outs_val, fid, -1)

if __name__ == '__main__':
    main()
