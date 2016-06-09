
This repo contains an implementation of the SegNet model using neon.

See the following references for more information:

```
"Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding."
Alex Kendall, Vijay Badrinarayanan and Roberto Cipolla
arXiv preprint arXiv:1511.02680, 2015.
```
[http://arxiv.org/abs/1511.02680](http://arxiv.org/abs/1511.02680)

```
"SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation."
arXiv preprint arXiv:1511.00561, 2015. 
Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla
```
[http://arxiv.org/abs/1511.00561](http://arxiv.org/abs/1511.00561)

## Data

Before running this model the data needs to be downloaded and converted into a format that
neon can use.  Soon neon will have optimized support for pixel-wise classification data,
but now the data must be stored in an HDF5 file (or other suitable format) so it can be
loaded using the ArrayIterator data iteration class.

### Downloading the dataset

This example uses the CamVid dataset which can be downloaded from the 
[SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial) repository.

The following code will need the path to the `CamVid` directory in the SegNet-Tutorial
directory; from now on denoted by CAMVID_PATH.


### Save data in HDF5 format

While the optimized dataloader for pixel-wise segmented images is still under development,
the CamVid images and pixelwise classification "images" need to be saved into a numpy array
for easy loading into the ArrayIterator neon data iterator object.  Other formats can be used,
as long as the input and target output data can be loaded into pythoin as numpy arrays.  See
the segment of the segnet_neon.py script where the train_set and val_set ArrayIterator objects
are instantiated.

The script to generate the HDF5 files with images is `gen_camvid_hdf5.py`.  This script goes through
the CamVid training, validation and test data set images and generates an HDF file with the arrays
in the format that can be easily used with neon.  To use the script, first enter the path to the
SegNet-Tutorial repo CamVid directory where the "pth_to_camvid" variable is initialized.
The script will generate hdf5 files in the CamVid directory.

The HDF5 file is not optimal in terms of file size, but these issues will be addressed soon with
enhancements to the neon dataloader.

This script requires the scipy python package to be installed.

## neon

Install neon by following the instructions at the neon repository
([https://github.com/NervanaSystems/neon](https://github.com/NervanaSystems/neon)).
The default installation uses a virtualenv, make sure to activate the virtual
env before running the SegNet scripts.  See neon documentation for more information.

This model has been tested with neon version tag [v1.4.0](https://github.com/NervanaSystems/neon/tree/v1.4.0), 
it may not work with other releases.


### SegNet implementation

The "segnet_neon.py" script is the main script to run SegNet using neon.  The model includes a pixelwise
softmax layer and the upsampling layer that is not included with the current
neon release.  These layers are defined in the files [upsampling_layer.py](./upsampling_layer.py) and
[pixelwise_sm.py](./pixelwise_sm.py), also the GPU kernels for the upsampling layer are included in
the file [segnet_neon_backend.py](./segnet_neon_backend.py).

To use this script, first edit the "pth_to_camvid" initialization to point to the CamVid directory
where the HDF5 data files are stored.

To fit the model use the command:
```
python segnet_neon.py -s <path to save checkpoints> -e 800 --serialize 10 -H 60 -r 1 -vvv
```
This will output a pickle file called `outputs.pkl` with the segmentation predictions
from the trained model on the test and validation sets.  The model state will be saved
every 10 epochs and 60 previous states will be retained.  The model will fit for 800
epochs.

After fitting, the `segnet_neon.py` script will run inference on the images in the test and 
validation sets and save the results to a pickle file named `outputs.pkl`.  The script in
[./check_outputs.py] shows how to view these results.  Also the script [./inference_example.py]
shows how to load the trained weights into neon and run inference on a PNG image file.  To
run this script you will need to give it the path to the SegNet serialized weight file and
the path to an image file.  Note that currently the model layers need to be regenerated to
load the trained weights, the model can not be deserialized directly from the serialized file
for this model.

To benchmark the model use the command:
```
python segnet_neon.py --bench
```

### Limitations

The upsampling layer implementation has only been tested with noa-noverlapping 2x2 kernels. To make sure the
pooling and upsampling pairs size the network properly and the pooling layer max pixel indicies are applied
correctly by the upsampling layer, only image with dimensions that are powers of two are supported.  Also
the image dimensions need to be larger than 2^N where N is total downsampling of the input image (i.e.
N is the number of 2x2 pooling layers in the encoder of the model).

There is no CPU backend support for this model, it requires a Maxwell class GPU.
