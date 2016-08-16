from neon import NervanaObject
import numpy as np
from itertools import izip
#from neon.data import DataLoader

class DepthImageLoader(NervanaObject):
    """
    """
    def __init__(self, dataloaders, shape):
        self.dataloaders = dataloaders
        self.shape = shape

        self.shapes = [ dataloaders[0].shape, dataloaders[1].shape]
        assert self.shapes[0] == self.shapes[1]

        self.out_shape = (self.shapes[0][0]*2, self.shapes[0][1], self.shapes[0][2])
        self.inp_full = self.be.iobuf(int(np.prod(self.out_shape)))
        self.nchan = self.out_shape[0]
        self.inp_full_vw = self.inp_full.reshape((self.nchan, -1))
        self.shape = self.out_shape
        self.output_f32 = self.be.iobuf(int(np.prod(self.out_shape[1:])))

        #self.t_int32 = None
        #self.tview = None

    @property
    def nbatches(self):
        return self.dataloaders[0].nbatches

    @property
    def ndata(self):
        return self.dataloaders[0].ndata

    def __iter__(self):
        for ((x1, t1), (x2, t2)) in izip(self.dataloaders[0], self.dataloaders[1]):
            self.inp_full_vw[0:3, :] = x1.reshape((3,-1))
            self.inp_full_vw[3:6, :] = x2.reshape((3,-1))
            self.output_f32[:] = t1
            yield (self.inp_full, self.output_f32)

    def reset(self):
        for dl in self.dataloaders:
            dl.reset()

    #def next(self, start):
    #    # first get the images from the dataloader for this minibatch
    #    x1, t  = .next(start)
    #    x2, t  = super(DepthImageLoader, self).next(start)
    #    if self.out_full is None:
    #        # generate the extra buffers needed to hold the expanded
    #        # one-hot target output
    #        self.out_full = self.be.iobuf(self.nclasses*t.shape[0], dtype=np.int32)
    #        self.outview = self.out_full.reshape((self.nclasses, -1))
#
    #        # need an int32 copy of the target classes for one-hot to work correctly
    #        self.t_int32 = self.be.iobuf(t.shape[0], dtype=np.int32)
    #        self.tview = self.t_int32.reshape((1, -1))
    #    # copy to int32
    #    self.t_int32[:] = t 
    #    # use backend one-hot to expand output to nclasses x H x W
    #    self.be.onehot(self.tview, axis=0, out=self.outview)
    #    return (x, self.out_full)

