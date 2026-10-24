#include "../../nvidia/common.cuh"
#include "rms_norm_nvidia.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

// One thread normalizes one row of X[M, K]; weight has K elements.
template <int MODE>
__global__ void rms_k(void *o, const void *x, const void *w, size_t M, size_t K, float eps) {
    size_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M) {
        float s = 0;
        for (size_t k = 0; k < K; k++) {
            float v = load<MODE>(x, r * K + k);
            s += v * v;
        }
        float inv = rsqrtf(s / K + eps);
        for (size_t k = 0; k < K; k++) {
            store<MODE>(o, r * K + k, load<MODE>(x, r * K + k) * load<MODE>(w, k) * inv);
        }
    }
}

} // namespace

void rms_norm(tensor_t o, tensor_t x, tensor_t w, float e) {
    size_t K = x->shape().back(), M = x->numel() / K;
    if (x->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((rms_k<0>), M, o->data(), x->data(), w->data(), M, K, e);
    } else if (x->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((rms_k<1>), M, o->data(), x->data(), w->data(), M, K, e);
    } else {
        LLAISYS_CUDA_LAUNCH((rms_k<2>), M, o->data(), x->data(), w->data(), M, K, e);
    }
}

} // namespace llaisys::ops::nvidia
