#
# example on how to check the inference outputs
# saved to the pickle file outputs.pkl by the
# segnet_neon.py script 
#

import pickle

# load the inference outputs from the pickle file
with open('outputs.pkl', 'r') as fid:
     (valout, testout) = pickle.load(fid)

# open the h5 files with the validation and test images
import os
import h5py
pth_to_camvid = 'SegNet-Tutorial/CamVid'  # or wherever the image hdf5 files were generated
with h5py.File(os.path.join(pth_to_camvid, 'val_images.h5'), 'r') as fid:
    shp = list(fid['input'].attrs['lshape'])
    mns = fid['input'].attrs['mean']
    shp.insert(0, -1)
    # reshape to be (N, H, W, C)
    val_imgs = np.array(fid['input']).reshape(shp).transpose((0,2,3,1))

    # add the mean value back in for each channel
    for ind in range(3):
        val_imgs[:,:,:,ind] += mns[ind]


import numpy as np
import matplotlib.pyplot as plt

# plot the first input image
plt.imshow(val_imgs[0]/256)

# get the pixelwise classification from the softmax SegNet outputs
img_segment = np.argmax(valout[0].reshape((12, 256, 512)), axis=0)

plt.figure()
plt.imshow(img_segment)
