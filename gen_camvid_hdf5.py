# load the camvid data sets and save in a ndarray, 
# output to a pickle file for use witht he ArrayIterator
import argparse
import os
import pickle
import numpy as np
from glob import glob
from scipy.misc import imresize
from scipy.ndimage import imread

import h5py

np.random.seed(1)


pth_to_camvid = 'SegNet-Tutorial/CamVid'

pth_to_camvid = os.path.abspath(pth_to_camvid)
if not os.path.isdir(pth_to_camvid):
    raise ValueError('Set path to SegNet-Tutorial repo CamVid dir')

H = 256
W = 512
C = 3
NCLASS = 12

new_shape = (C, H, W)
new_shape_annot = (NCLASS, H, W)
mns = np.zeros((C,))
for settype in ['train', 'val', 'test']:
    print settype

    # load up the image and the corresponding annotation
    fns = glob(os.path.join(pth_to_camvid, settype, '*.png'))
    # randomize image order
    inds = np.random.permutation(len(fns))

    imgs = np.zeros((len(fns), np.prod(new_shape)))
    imgs_annot = np.zeros((len(fns), np.prod(new_shape_annot)))
    im_annot_out = np.zeros(new_shape_annot)

    # the settype + 'files' file in the CamVid directory will
    # have the output files in order
    fid2 = open(os.path.join(pth_to_camvid, settype + 'files'), 'w')
    for cnt in range(len(fns)):
        fn = fns[inds[cnt]]
        fid2.write(fn + '\n')

        # load the image
        im = imread(fn)
        fn_annot = os.path.join(pth_to_camvid, settype + 'annot', os.path.basename(fn))
        im_annot = imread(fn_annot)

        # resize images to 256x512
        im = imresize(im, new_shape[1:3])
        im = im.transpose((2,0,1))

        im_annot = imresize(im_annot, new_shape_annot[1:3], interp='nearest')

        # map the im_annot to channel
        for h in range(new_shape_annot[1]):
            for w in range(new_shape_annot[2]):
                chan_ind = int(im_annot[h, w])
                im_annot_out[chan_ind, h, w] = 1.0

        if settype == 'train':
            mns += np.mean(np.mean(im, axis=2), axis=1)

        # Nimages x (flattened shape)
        imgs[cnt, :] = im.flatten()
        imgs_annot[cnt, :] = im_annot_out.flatten()

    fid2.close()

    # calculate the mean pixel vals for each channel
    # for the training set
    if settype == 'train':
        mns /= float(cnt)
        mns_array = np.zeros(new_shape).transpose((1,2,0))
        mns_array[:] = mns
        mns_array = mns_array.transpose((2,0,1)).flatten().copy()

    imgs = imgs - mns_array

    with h5py.File(os.path.join(pth_to_camvid, settype + '_images.h5'), mode='w') as out_file:
        out_file.create_dataset("input", data=imgs.astype(np.float32))
        out_file['input'].attrs['lshape'] = new_shape
        out_file['input'].attrs['mean'] = mns
        out_file.create_dataset("output", data=imgs_annot.astype(np.uint8), dtype='uint8')
        out_file['output'].attrs['lshape'] = new_shape_annot
