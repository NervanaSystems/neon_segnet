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
import matplotlib.pyplot as plt

from neon import NervanaObject
from neon.backends import gen_backend
from neon.models import Model
from neon.data import ImageParams
from neon.models import Model
from neon.util.persist import load_obj
from neon.util.modeldesc import ModelDescription

from segnet_neon_backend import NervanaGPU_Upsample
from segnet_neon_backend import _get_bprop_upsampling
from pixelwise_dataloader import PixelWiseImageLoader
from segnet_neon import gen_model
global _get_bprop_upsampling

# need to use the backend with the new upsampling layer implementation
be = NervanaGPU_Upsample(device_id=0)

# set batch size
be.bsz = 1

# couple backend to global neon object
NervanaObject.be = be

shape = dict(channel_count=3, height=256, width=512, subtract_mean=False)
test_params = ImageParams(center=True, flip=False,
                          scale_min=256, scale_max=256,
                          aspect_ratio=0, **shape)
common = dict(target_size=256*512, target_conversion='read_contents',
              onehot=False, target_dtype=np.uint8, nclasses=12)
data_dir = '/usr/local/data/evren/CamVid/'

test_set = PixelWiseImageLoader(set_name='test', repo_dir=data_dir,media_params=test_params, 
                                index_file=os.path.join(data_dir, 'test_images.csv'), **common)


# initialize model object
segnet_model = Model(layers=gen_model())
segnet_model.initialize(test_set)

# load up the serialized model
model_desc = ModelDescription(load_obj('testmodel.prm'))
for layer in segnet_model.layers_to_optimize:
    name = layer.name
    trained_layer = model_desc.getlayer(name)
    layer.load_weights(trained_layer)

plt.figure()

for x, t in test_set:
    z = segnet_model.fprop(x).get()

    z = np.argmax(z.reshape((12, 256, 512)), axis=0)
    t = np.argmax(t.get().reshape((12, 256, 512)), axis=0)
    plt.subplot(2,1,1);plt.gca().cla();plt.imshow(t);plt.title('Truth')
    plt.subplot(2,1,2);plt.gca().cla();plt.imshow(z);plt.title('SegNet')
    plt.show()
    raw_input()
    

    




