import numpy as np
from neon.data import DataLoader

class PixelWiseImageLoader(DataLoader):
    """
    Expand the normal dataloader to take the output pixelwise image
    and expand it to to a (nclass, H, W) pixelwise one hot output.
    The dataloader expects the target images for pixelwise classification
    to be HxW and for the pixel value to be an integer specifying the
    output class of the image.  For pixelwise classification using softmax
    activation the target image needs to be expanded to a one-hot representation
    where the target image is a (nclass, H, W) tensor.  For each pixel, there
    is a single '1' in the first channel/dimension of the tensor for the
    true output class label for that pixel.
    """
    def __init__(self, *args, **kwargs):
        super(PixelWiseImageLoader, self).__init__(*args, **kwargs)
        self.out_full = None
        self.t_int32 = None
        self.tview = None

    def next(self, start):
        # first get the images from the dataloader for this minibatch
        x, t  = super(PixelWiseImageLoader, self).next(start)
        if self.out_full is None:
            # generate the extra buffers needed to hold the expanded
            # one-hot target output
            self.out_full = self.be.iobuf(self.nclasses*t.shape[0], dtype=np.int32)
            self.outview = self.out_full.reshape((self.nclasses, -1))

            # need an int32 copy of the target classes for one-hot to work correctly
            self.t_int32 = self.be.iobuf(t.shape[0], dtype=np.int32)
            self.tview = self.t_int32.reshape((1, -1))
        # copy to int32
        self.t_int32[:] = t 
        # use backend one-hot to expand output to nclasses x H x W
        self.be.onehot(self.tview, axis=0, out=self.outview)
        return (x, self.out_full)

