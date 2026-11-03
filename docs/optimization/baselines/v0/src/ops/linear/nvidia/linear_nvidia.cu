#include "../../nvidia/common.cuh"
#include "linear_nvidia.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

// X[M, K] * W[N, K]^T + bias[N] -> output[M, N].
// One thread computes one output element, accumulating across K in float.
template <int MODE>
__global__ void linear_k(void *o, const void *x, const void *w, const void *bias, size_t M,
                         size_t N, size_t K) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * N) {
        size_t m = i / N, n = i % N;
        float s = bias ? load<MODE>(bias, n) : 0.f;
        for (size_t k = 0; k < K; k++) {
            s += load<MODE>(x, m * K + k) * load<MODE>(w, n * K + k);
        }
        store<MODE>(o, i, s);
    }
}

} // namespace

void linear(tensor_t o, tensor_t x, tensor_t w, tensor_t bias) {
    size_t K = x->shape().back(), M = x->numel() / K, N = w->shape()[0];
    if (w->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((linear_k<0>), o->numel(), o->data(), x->data(), w->data(),
                            bias ? bias->data() : nullptr, M, N, K);
    } else if (w->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((linear_k<1>), o->numel(), o->data(), x->data(), w->data(),
                            bias ? bias->data() : nullptr, M, N, K);
    } else {
        LLAISYS_CUDA_LAUNCH((linear_k<2>), o->numel(), o->data(), x->data(), w->data(),
                            bias ? bias->data() : nullptr, M, N, K);
    }
}

} // namespace llaisys::ops::nvidia
