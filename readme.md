
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

## Requirements

To run these models [neon v1.5.3](https://github.com/NervanaSystems/neon/tree/v1.5.3)
must be installed on the system.  Install neon by following the instructions at the
neon repository ([https://github.com/NervanaSystems/neon](https://github.com/NervanaSystems/neon)).
The default installation uses a virtualenv, make sure to activate the virtual
env before running the SegNet scripts.  See neon documentation for more information.


Also, addtional python package requirements specified in the
[requirements.txt](./requirements.txt) file must be installed using the command:
```
# if neon is installed in a venv
# first activate the venv
source <path to neon>/.venv/bin/activate

pip install-r requirements.txt
```

## Data

Before running this model the data needs to be downloaded and converted into a format that
neon can use.  

### Downloading the dataset

This example uses the CamVid dataset which can be downloaded from the 
[SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial) repository.

The following code will need the path to the `CamVid` directory in the SegNet-Tutorial
directory; from now on denoted by CAMVID_PATH.

### Converting dataset for use with neon

Currently the neon version of SegNet requires that the images be powers of 2 in each
dimension.  For this example, the CamVid 360x480 color images are reshaped to 256x512,
as are the target output images.  To generate the neon compatible images, use the
script [proc_images.py](./proc_images.py).  Make sure to install the python packages
included in the [requirements.txt](./requirements.txt) before running this script.
The script requires the path to the CamVid directory in the SegNet-Tutorial repo that
was cloned and the output path to place the neon compatible PNG files.  For example:

```
python proc_images.py /path/to/SegNet-Tutorial/CamVid/ /local/dir/CamVid_neon/
```

The output dir will contain the directories train, test and val with the input training,
testing and validation image sets, repectively.  Also, for each set there will be a
corresponding directory with 'annot' added to the basename holding the annotation
images.  The annotation images and grayscale PNG files where each pixel holds an interger
from 0 to 12.  The pixcel value indicates the ground truth output class of that pixel.

### SegNet implementation

The "segnet_neon.py" script is the main script to run SegNet using neon.
The model includes a pixelwise softmax layer and the upsampling layer
that is not included with the current neon release.  Also, a special data loader
class is included which converts the 1 channel target class images holding
the groudn truth values for each pixel into a 12 channel image using a one-hot
representation for the class of each pixel.  This is required for the logistic
regression using neon.

These layers and the data loader are defined in the files
[upsampling_layer.py](./upsampling_layer.py),
[pixelwise_dataloader.py]./pixelwise_dataloader.py()
and [pixelwise_sm.py](./pixelwise_sm.py). Also the GPU kernels for the upsampling
layer are included in the file [segnet_neon_backend.py](./segnet_neon_backend.py).

To fit the model use the command such as:
```
python segnet_neon.py -s <path to save checkpoints> -e 650 --serialize 10 -z 4 \
                      -H 60 -r 1 -vvv -eval 1 -w /path/to/processed/CamVid/
```
The path to the data generated using the [proc_images.py](./proc_images.py) 
msut be provided.  The '-z 4' option sets the batch size to 4, this is close
to the maximum batch size for a GPU with 12GB of memory.

During training, neon will output the total sum of the multiclass cross entropy
function over ever pixel.  These values can be large since they are not normalized
or averaged over the number of pxiels in the image.  Also, at the end of every epoch,
the cross entropy of over the validation set will be printed.

After fitting, the `segnet_neon.py` script will run inference on the images in the test and 
validation sets and save the results to a pickle file named `outputs.pkl`.  The script in
[./check_outputs.py] shows how to view these results.  Also the script [./inference_example.py]
shows how to load the trained weights into neon and run inference on a PNG image file.  To
run this script you will need to give it the path to the SegNet serialized weight file and
the path to an image file.  Note that currently the model layers need to be regenerated to
load the trained weights, the model can not be deserialized directly from the serialized file
for this model.


### Benchmarks

To benchmark the model use the command:
```
python segnet_neon.py --bench
```

Machine and GPU specs:
```
Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz
Ubuntu 14.04.2 LTS
GPU: GeForce GTX TITAN X
CUDA Driver Version 7.0
```

Runtimes:
```
-----------------------------
|    Func     |    Mean     |
-----------------------------
| fprop       |  108.95  ms |
| bprop       |  193.27  ms |
 ------------- -------------
| iteration   |  302.22  ms |
-----------------------------
```


### Limitations

The upsampling layer implementation has only been tested with non-noverlapping
2x2 kernels. To make sure the pooling and upsampling pairs size the network properly
and the pooling layer max pixel indicies are applied correctly by the upsampling layer,
only image with dimensions that are powers of two are supported.  Also the image
dimensions need to be larger than 2^N where N is total downsampling of the input
image (i.e.  N is the number of 2x2 pooling layers in the encoder of the model).

There is no CPU backend support for this model, it requires a Maxwell class GPU.
