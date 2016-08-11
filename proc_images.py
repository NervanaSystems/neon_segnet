"""
Script to generate the CSV files from the the CamVid data set
which are used by the neon dataloader to load the images and
annotations into the neon model.

Arguments:
    image_path (str): path to the CamVid data set found in the SegNet-Tutorial
                      GitHub repo (https://github.com/alexgkendall/SegNet-Tutorial)
                      this path should have the directories: 'train', 'trainannot',
                      'val', 'valannot', 'test', 'testannot'
    output_path (str): path to place the image files to be used by neon, it is best
                       to use a path on the local drive instead of a network mouonted drive
"""

import os
import argparse
from glob import glob
from scipy.ndimage import imread
from scipy.misc import imresize, imsave
parser = argparse.ArgumentParser(description='generate csv file from CamVid data set')
parser.add_argument('image_path', type=str, help='path to the CamVid data set in the SegNet-Tutorial '
                                           'GitHub repo')
parser.add_argument('output_path', type=str, help='directory to store the neon compatible images')
args = parser.parse_args()

def main():
    assert os.path.isdir(args.image_path), '%s directory not found' % args.mage_path

    for dataset in ['train', 'test', 'val']:
        out_dir_im = os.path.join(args.output_path, dataset)
        if not os.path.isdir(out_dir_im):
            os.makedirs(out_dir_im)

        out_dir_an = os.path.join(args.output_path, dataset + 'annot')
        if not os.path.isdir(out_dir_an):
            os.makedirs(out_dir_an)

        fid = open(os.path.join(args.output_path, '%s_images.csv' % dataset), 'w')
        # print header
        fns = glob(os.path.join(args.image_path, dataset, '*.png'))

        for fn in fns:

            fn_image = os.path.abspath(fn)
            fn_annot = os.path.split(fn_image)
            fn_annot = os.path.join(fn_annot[0] + 'annot', fn_annot[1])

            im = imread(fn_image)
            annot = imread(fn_annot)
            out_size = (256, 512)
            im = imresize(im, out_size)
            annot = imresize(annot, out_size, interp='nearest')


            fn_image_out = os.path.abspath(os.path.join(out_dir_im,
                                                        os.path.basename(fn_image)))
            fn_annot_out = os.path.abspath(os.path.join(out_dir_an,
                                                        os.path.basename(fn_image)))
            imsave(fn_image_out, im)
            imsave(fn_annot_out, annot)

            fid.write('%s,%s\n' %(fn_image_out, fn_annot_out))
        fid.close()

if __name__ == '__main__':
    main()
