import numpy as np

from neon import NervanaObject
from neon.backends.backend import Tensor
from neon.layers.layer import Layer

class Upsampling(Layer):
    """
    Upsampling layer implementation.  Layer uses the argmax
    indicies from a upstream pooling layer to make a sparse
    upsampling of the input data.  Current only a 2x upsampling
    is supported.

    For reference see:
    "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation"
    Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla
    http://arxiv.org/pdf/1511.00561v2.pdf

    Arguments:
        fshape (int, (int, int)): one or two dimensional shape
                                  of upsampling window
        pait_layer (Pooling): The upstream pooling layer object for which
                              this layer is unpooling for.  The argmax values
                              from this layer will be used for upsampling
        strides (int, dict, optional): strides to apply upsampling window
            over. An int applies to both dimensions, or a dict with str_h
            and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        padding (int, dict, optional): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        name (str, optional): layer name. Defaults to "Upsampling"
    """

    def __init__(self, fshape, pair_layer, strides={}, padding={},
                 name=None):
        super(Upsampling, self).__init__(name)
        self.poolparams = {'str_h': None, 'str_w': None, 'str_d': None,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0, 
                           'op': 'max'}  # 3D paramaters

        if type(fshape) is int:
            fshape = (fshape, fshape)
        assert fshape == (2, 2), 'Only 2x2 kernels currently supported'

        if type(strides) is int:
            assert strides == 2, 'Only stride of 2 is supported currently'
        else:
            assert strides['str_h'] == 2 and strides['str_w'] == 2, \
                    'Only stride of 2 is supported currently'

        if type(padding) is int:
            assert padding == 0, 'Only padding of 0 is supported currently'
        else:
            assert padding['pad_h'] == 2 and padding['pad_w'] == 2, \
                    'Only padding of 2 is supported currently'

        self.op = 'max'
        self.fshape = fshape
        self.strides = strides
        self.padding = padding
        self.owns_delta = True
        if isinstance(fshape, int):
            fshape = {'R': fshape, 'S': fshape}
        elif isinstance(fshape, tuple):
            fkeys = ('R', 'S') if len(fshape) == 2 else ('T', 'R', 'S')
            fshape = {k: x for k, x in zip(fkeys, fshape)}
        elif fshape == 'all':
            raise ValueError('not supported yet')
            fshape = dict(R=None, S=None)
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding}
        for d in [fshape, strides, padding]:
            self.poolparams.update(d)
        self.nglayer = None
        self.pair_layer = pair_layer

    def get_description(self, **kwargs):
        if 'skip' in kwargs:
            kwargs['skip'].append('pair_layer')
        else:
            kwargs['skip'] = ['pair_layer']
        return super(Upsampling, self).get_description(**kwargs)

    def __str__(self):
        strout = "upsampling layer '%s': %d x (%dx%d) inputs, %d x (%dx%d) outputs"
        return strout % (self.name, self.in_shape[0], self.in_shape[1], self.in_shape[2],
                         self.out_shape[0], self.out_shape[1], self.out_shape[2])

    def configure(self, in_obj):
        super(Upsampling, self).configure(in_obj)
        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            assert len(self.in_shape) == 3, 'no 3d support'
            ikeys = ('C', 'P', 'Q')
            shapedict = {k: x for k, x in zip(ikeys, self.in_shape)}
            shapedict['N'] = self.be.bsz
            self.poolparams.update(shapedict)
            if self.poolparams['R'] is None:
                self.poolparams['R'] = shapedict['H']
                self.poolparams['S'] = shapedict['W']
            self.nglayer = self.be.upsampling_layer(self.be.default_dtype,
                                                    **self.poolparams)
            self.out_shape = (self.nglayer.C, self.nglayer.H, self.nglayer.W)
        return self

    def set_deltas(self, delta_buffers):
        super(Upsampling, self).set_deltas(delta_buffers)
        self.argmax = self.pair_layer.argmax

    def fprop(self, inputs, inference=False, beta=0.0):
        self.inputs = inputs
        self.be.bprop_pool(self.nglayer, inputs, self.outputs, self.argmax, alpha=1., beta=0.0)
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        self.be.fprop_pool(self.nglayer, error, self.deltas, self.argmax, beta=beta)
        return self.deltas
