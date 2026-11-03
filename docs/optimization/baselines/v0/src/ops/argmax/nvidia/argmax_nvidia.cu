#include "../../nvidia/common.cuh"
#include "argmax_nvidia.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

// Reduce each row of length K; strict > preserves the first index on ties.
template <int MODE>
__global__ void argmax_k(int64_t *idx, void *val, const void *x, size_t rows, size_t K) {
    size_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float best = load<MODE>(x, r * K), v;
        size_t bi = 0;
        for (size_t k = 1; k < K; k++) {
            v = load<MODE>(x, r * K + k);
            if (v > best) {
                best = v;
                bi = k;
            }
        }
        idx[r] = bi;
        store<MODE>(val, r, best);
    }
}

} // namespace

void argmax(tensor_t idx, tensor_t val, tensor_t x) {
    size_t K = x->shape().back(), rows = x->numel() / K;
    if (x->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((argmax_k<0>), rows, reinterpret_cast<int64_t *>(idx->data()),
                            val->data(), x->data(), rows, K);
    } else if (x->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((argmax_k<1>), rows, reinterpret_cast<int64_t *>(idx->data()),
                            val->data(), x->data(), rows, K);
    } else {
        LLAISYS_CUDA_LAUNCH((argmax_k<2>), rows, reinterpret_cast<int64_t *>(idx->data()),
                            val->data(), x->data(), rows, K);
    }
}

} // namespace llaisys::ops::nvidia
