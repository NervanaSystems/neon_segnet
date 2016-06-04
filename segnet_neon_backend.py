"""
Backend functions needed for the Upsampling layers in the SegNet Model
"""
from operator import mul

from pycuda.compiler import SourceModule
from pycuda.tools import context_dependent_memoize
from neon.backends import cuda_templates

from neon.backends.cuda_templates import (_common_fp16_to_fp32,
                                          _common_round,  # for fp32_to_fp16 converter
                                          _common_max_abs,
                                          _common_kepler,
                                          _ew_types)
from neon.backends.kernels.cuda.pooling import prepare_template_vals

# This section of the code contains templated CUDA-C code for the kernels.
@context_dependent_memoize
def _get_bprop_upsampling(clss, compute_capability):

    code = r"""
#define FLT_MAX 3.402823466E+38F

%(common)s

__global__ void spool_bprop_upsampling(
    const %(type)s* I, %(type)s* O, unsigned char* A,
    float alpha, float beta, int flags,
    int N, int W, int H, int D, int C,
    int WN, int HWN, int DHWN, int P, int Q,
    int magic_P, int shift_P, int QN, int PQN, int MPQN,
    int pad_c, int pad_d, int pad_h, int pad_w,
    int str_c, int str_d, int str_h, int str_w,
    int S, int RS, int RST, int JRST,
    int magic_S, int shift_S,
    int magic_RS, int shift_RS, int magic_RST, int shift_RST,
    int supP, int supQ, int shlP, int maskP, int shrP,
    int shlQ, int maskQ, int shrQ, int maskN, int shrN
    %(stats_args)s
    )
{
    extern __shared__ int lut[];
    int tid = threadIdx.x;

    int q  = blockIdx.x;
    int mp = blockIdx.y;
    int k  = blockIdx.z;

    int m = mp * magic_P; m >>= shift_P;
    int p = mp - m*supP;

    // zigzag q back and forth to improve L2 cache perf
    if (p & 1)
        q = supQ - q - 1;

    // Superblock P and Q
    p = (p << shlP) + ((tid & maskP) >> shrP);
    q = (q << shlQ) + ((tid & maskQ) >> shrQ);
    int n = tid & maskN;

    int sb = tid >> shrN;

    int offset = k*MPQN + m*PQN + p*QN + mad16(q, N, n);
    I += n;
    O += offset;
    A += offset;

    float O_val = beta != 0.0f && p < P && q < Q && n < N ? %(cvt)s(__ldg(O)) : 0.0f;

    if (tid < 32)
    {
        int kj = k * str_c - pad_c;
        int mt = m * str_d - pad_d;
        int pr = p * str_h - pad_h;
        int qs = q * str_w - pad_w;

        int inc = min(maskN + 1, 32);

        int jrst = n;
        while (jrst < JRST)
        {
            int j   = div16(jrst, magic_RST, shift_RST);
            int rst = mod16(jrst, j, RST);

            int t   = div16(rst, magic_RS, shift_RS);
            int rs  = mod16(rst, t, RS);

            int r   = div16(rs, magic_S, shift_S);
            int s   = mod16(rs, r, S);

            int x = qs + s;
            int y = pr + r;
            int z = mt + t;
            int c = kj + j;

            bool bounds_x  = x >= 0 && x < W;
            bool bounds_y  = y >= 0 && y < H;
            bool bounds_z  = z >= 0 && z < D;
            bool bounds_c  = c >= 0 && c < C;
            bool in_bounds = bounds_x && bounds_y && bounds_z && bounds_c;

            int sliceI  = c*DHWN + z*HWN + y*WN + x*N;

            int lut_offset = mad16(sb, JRST, jrst);

            lut[lut_offset] = in_bounds ? sliceI : -1;
            jrst += inc;
        }
    }
    __syncthreads();

    int intermediate_max = 0;

    if (p < P && q < Q && n < N)
    {
        int jrst = 0;
        int argmax = 0;
        float max = -FLT_MAX;
        while (jrst < JRST)
        {
            int lut_offset = mad16(sb, JRST, jrst);

            //int slice0 = lut[lut_offset + 0];
            //int slice1 = lut[lut_offset + 1];
            //int slice2 = lut[lut_offset + 2];
            //int slice3 = lut[lut_offset + 3];
            int slice = lut[lut_offset + *A - jrst];

            // val needs to stay in fp32 or can't be se to FLT_MAX
            //float val0 = jrst + 0 < JRST && slice0 >= 0 ? %(cvt)s(__ldg(I + slice0)) : -FLT_MAX;
            //float val1 = jrst + 1 < JRST && slice1 >= 0 ? %(cvt)s(__ldg(I + slice1)) : -FLT_MAX;
            //float val2 = jrst + 2 < JRST && slice2 >= 0 ? %(cvt)s(__ldg(I + slice2)) : -FLT_MAX;
            //float val3 = jrst + 3 < JRST && slice3 >= 0 ? %(cvt)s(__ldg(I + slice3)) : -FLT_MAX;

            //if (*A == jrst + 0) {
            //    max = val0;
            //}
            //if (*A == jrst + 1) {
            //    max = val1;
            //}
            //if (*A == jrst + 2) {
            //    max = val2;
            //}
            //if (*A == jrst + 3) {
            //    max = val3;
            //}

            max = %(cvt)s(__ldg(I + slice));

            jrst += 4;
        }
        // convert back to fp to write out
        %(type)s temp_out = %(cvt_out)s( %(mul_by_scale)s (max*alpha + O_val*beta));
        if (!(flags & 1)) {
            *O = temp_out;
        }

        intermediate_max = max_abs(0, temp_out);  // compute abs
    }
    intermediate_max += 0;
    %(atomic_max)s
}
"""

    template_vals = prepare_template_vals(clss, compute_capability)
    code = code % template_vals
    module = SourceModule(code)
    kernel = module.get_function("spool_bprop_upsampling")
    sig = "3P 2f 44I" + ("Pf" if (clss[0] == "x") else "")
    kernel.prepare(sig)
    return kernel

from neon.backends.layer_gpu import PoolLayer
class UpsamplingLayer(PoolLayer):
    def __init__(self, lib, dtype,
                 op, N, C,
                 P, Q,
                 R=1, S=1, 
                 pad_c=0, pad_d=0, pad_h=0, pad_w=0,
                 str_c=None, str_d=None, str_h=None, str_w=None):

        assert op == 'max'
        assert str_h == 2 and str_w == 2, 'Currently only support stride 2 for upsampling'
        assert R == 2 and S == 2, 'Currently only support fshape 2 for upsampling'

        D=T=1
        J=1  
        H = (P-1) * str_h - 2 * pad_h + R
        W = (Q-1) * str_w - 2 * pad_w + S

        super(UpsamplingLayer, self).__init__(lib, dtype, 
                 op, N, C,
                 D, H, W,
                 J, T, R, S,
                 pad_c, pad_d, pad_h, pad_w,
                 str_c, str_d, str_h, str_w)

        self.fprop_kernel[0] = 'bprop_upsampling'

        self.H = H
        self.W = W
        self.nOut = reduce(mul, self.DHW, 1) * C


from neon.backends.nervanagpu import NervanaGPU
class NervanaGPU_Upsample(NervanaGPU):
    def upsampling_layer(self, dtype, op, N, C,
                         P, Q, R=2, S=2,
                         pad_d=0, pad_h=0, pad_w=0,
                         str_d=1, str_h=2, str_w=2):
        assert str_h == 2 and str_w == 2, 'Currently only support stride 2 for upsampling'
        assert R == 2 and S == 2, 'Currently only support fshape 2 for upsampling'
        return UpsamplingLayer(self, dtype, op, N, C, P, Q, R, S,
                               pad_d=pad_d, pad_h=pad_h, pad_w=pad_w,
                               str_d=str_d, str_h=str_h, str_w=str_w)

    def _execute_pool(self, layer, I, O, argmax, alpha, beta, kernel_args, shared, repeat):
        if type(layer) is not UpsamplingLayer:
            super(NervanaGPU_Upsample, self)._execute_pool(layer, I, O, argmax,
                                                           alpha, beta, kernel_args,
                                                           shared, repeat)
            return

        assert I.dtype == O.dtype
        A_data = argmax.gpudata if argmax is not None else 0
        kernel = _get_bprop_upsampling(layer.dtype.str[1:], self.compute_capability)
        flags = 0
        params = [kernel_args[1], kernel_args[2], self.stream,
                  I.gpudata, O.gpudata, A_data, alpha, beta, flags]
        params.extend(kernel_args[3])

        for r in range(repeat):
            kernel.prepared_async_call(*params, shared_size=shared)
