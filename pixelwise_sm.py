
from neon.transforms.transform import Transform

class PixelwiseSoftmax(Transform):
    """
    Pixelwise SoftMax activation function.
    Computes the function f(x_k) = exp(x_k) / sum_i(exp(x_i))
    """
    def __init__(self, c, h, w, name=None, epsilon=2**-23):
        super(PixelwiseSoftmax, self).__init__(name)
        self.epsilon = epsilon
        self.w = w
        self.h = h
        self.c = c

    def __call__(self, x):
        #import pdb; pdb.set_trace()
        # size of x is (808*608*4, bsz)
        y = x.reshape((self.c, -1))
        y[:] = (self.be.reciprocal(self.be.sum(self.be.exp(y - self.be.max(y, axis=0)), axis=0)) * self.be.exp(y - self.be.max(y, axis=0)))
        return x

    def bprop(self, x):
        return 1
