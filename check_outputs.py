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
import argparse
import numpy as np

from neon import NervanaObject
from neon.backends import gen_backend
from neon.models import Model
from neon.util.persist import load_obj
from neon.util.modeldesc import ModelDescription

from segnet_neon_backend import NervanaGPU_Upsample
from segnet_neon_backend import _get_bprop_upsampling
from segnet_neon import gen_model
from segnet_neon import make_aeon_config, transformers
from aeon import DataLoader

global _get_bprop_upsampling

parser = argparse.ArgumentParser(description='compare output of trained model to ground truth')
parser.add_argument('image_path', type=str, help='path to the CamVid data set in the SegNet-Tutorial '
                                                 'GitHub repo')
parser.add_argument('save_model_file', type=str, help='serialized model file')
parser.add_argument('output_dir', type=str, help='directory to save the output images')
parser.add_argument("--num_classes", type=int, default=12, help="number of annotated classes")
parser.add_argument("--height", type=int, default=256, help="image height")
parser.add_argument("--width", type=int, default=512, help="image width")
parser.add_argument("--display", action="store_true", help="output to screen instead of to file")
parser.add_argument("--valset", action="store_true", help="Use the validation set instead of the test set")
args = parser.parse_args()
if not args.display:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

if not os.path.isdir(args.output_dir):
    print 'making output diriextory [%s]' % args.output_dir
    os.makedirs(args.output_dir)
else:
    print 'will put output images in directory %s' % args.output_dir

# check that image dimensions are powers of 2
if((args.height & (args.height - 1)) != 0):
    raise TypeError("Height must be a power of 2.")
if((args.width & (args.width - 1)) != 0):
    raise TypeError("Width must be a power of 2.")

(c, h, w) = (args.num_classes, args.height, args.width)

# need to use the backend with the new upsampling layer implementation
be = NervanaGPU_Upsample(device_id=0)

# set batch size
be.bsz = 1

# couple backend to global neon object
NervanaObject.be = be

if args.valset:
    manifest = os.path.join(args.image_path, 'val_images.csv')
else:
    manifest = os.path.join(args.image_path, 'test_images.csv')
dat_config = make_aeon_config(manifest, h, w, be.bsz)
dataloader = DataLoader(dat_config, be)
dataset = transformers(dataloader, c)

# initialize model object
segnet_model = Model(layers=gen_model(c, h, w))
segnet_model.initialize(dataset)

# load up the serialized model
model_desc = ModelDescription(load_obj(args.save_model_file))
for layer in segnet_model.layers_to_optimize:
    name = layer.name
    trained_layer = model_desc.getlayer(name)
    layer.load_weights(trained_layer)

fig = plt.figure()
if args.display:
    plt.ion()

im1 = None
im2 = None

cnt = 1
for x, t in dataset:
    z = segnet_model.fprop(x).get()

    z = np.argmax(z.reshape((c, h, w)), axis=0)
    t = np.argmax(t.get().reshape((c, h, w)), axis=0)

    # calculate the misclass rate
    acc = (np.where(z == t)[0].size / float(z.size))*100.0

    plt.subplot(2,1,1);
    if im1 is None:
        im1 = plt.imshow(t);plt.title('Truth')
    else:
        im1.set_data(t)

    plt.subplot(2,1,2);
    if im2 is None:
        im2 = plt.imshow(z);
    else:
        im2.set_data(z)
    plt.title('SegNet - pixelwise acc = %.1f %%' % acc )

    fig.savefig(os.path.join(args.output_dir, 'out_%d.png' % cnt))
    cnt += 1
    if args.display:
        plt.draw()
        plt.show()
        raw_input('Pixelwise classification accuracy = %.1f, press return to continue...' % acc)
