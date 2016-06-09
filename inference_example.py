#
# example showing how to load up the trained segnet model
# and run inference on a single PNG file
#

import os
from neon import NervanaObject
from neon.models import Model
from neon.util.persist import load_obj

from neon.initializers import  Kaiming
from neon.layers import Conv, GeneralizedCost, Dropout, Pooling
from neon.transforms import Rectlin

from segnet_neon_backend import _get_bprop_upsampling
from segnet_neon_backend import NervanaGPU_Upsample

from pixelwise_sm import PixelwiseSoftmax

from upsampling_layer import Upsampling
from segnet_neon_backend import _get_bprop_upsampling
global _get_bprop_upsampling


# currently deserialization of this model
# requires instantiating the layers again
be = NervanaGPU_Upsample(device_id=0,
                         cache_dir=os.path.join(os.path.expanduser('~'), 'nervana/cache'))
be.bsz = 1 

# set bactch size in globa backend object
NervanaObject.be = be

# generate the layers
init_uni = Kaiming()
conv_common = dict(padding=1, init=init_uni, activation=Rectlin(), batch_norm=True)
layers=[]
pool_layers = []

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

lshape = (12, 256, 512)
c, h, w = lshape
act_last = PixelwiseSoftmax(c, h, w, name="PixelwiseSoftmax")
conv_last = dict(padding=1, init=init_uni, activation=act_last, batch_norm=False)
layers.append(Conv((3, 3, c), strides=1, name='deconv1_1', **conv_last))


# instantiate the model object with these layers
segnet_model = Model(layers=layers)

# now load the params from the save model
trained_model_file = 'run1/segnet_train.prm'
segnet_model.deserialize(load_obj(trained_model_file), load_states=False)

# setup all the buffers
in_shape = (3, 256, 512)
segnet_model.initialize(in_shape)

# if using neon venv may need to install matplotlib and scipy
# into the venv or provide path to this libraries
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
import numpy as np

# load the image and resize to 
pth_to_image = 'SegNet-Tutorial/CamVid/val/0016E5_08127.png'
inp_im = imread(pth_to_image)
# resize to the shape expected by the model (256, 512)
inp_im = imresize(inp_im, (256, 512))

# plot the imaage
plt.imshow(inp_im)

# generate an input buffer on the GPU
# will be an array of shape (C*H*W, N)
# N is the batch size
inp_buf = be.iobuf(3*256*512)

#convert to float
inp_im = inp_im.astype(np.float32)
# subtract the channel means
mns = np.array([105.0387935 ,  108.41374745,  110.33613568]).astype(np.float32)
inp_im -= mns
# reorder the image dims to CHW
inp_im = inp_im.transpose((2,0,1))

# set the first image in the mini-batch to this image
inp_buf[:, 0] = inp_im.flatten()

# run fprop and get the outputs
out = segnet_model.fprop(inp_buf)

# pull output image to numpy array on host
# and unflatten
out = out.get().reshape(lshape)

# take argmax over the pixel categories
out_cat = np.argmax(out, axis=0)
plt.figure()
plt.imshow(out_cat)
