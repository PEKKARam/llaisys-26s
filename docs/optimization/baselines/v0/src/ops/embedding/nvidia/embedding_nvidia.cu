#include "../../nvidia/common.cuh"
#include "embedding_nvidia.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

// ids[n] selects rows from weight[vocab, d] into output[n, d].
template <int MODE>
__global__ void emb_k(void *o, const int64_t *ids, const void *w, size_t n, size_t d) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * d) {
        store<MODE>(o, i, load<MODE>(w, (size_t)ids[i / d] * d + i % d));
    }
}

} // namespace

void embedding(tensor_t o, tensor_t ids, tensor_t w) {
    int m = w->dtype() == LLAISYS_DTYPE_F32 ? 0 : w->dtype() == LLAISYS_DTYPE_BF16 ? 1 : 2;
    if (m == 0) {
        LLAISYS_CUDA_LAUNCH((emb_k<0>), o->numel(), o->data(),
                            reinterpret_cast<const int64_t *>(ids->data()), w->data(), ids->numel(),
                            w->shape()[1]);
    } else if (m == 1) {
        LLAISYS_CUDA_LAUNCH((emb_k<1>), o->numel(), o->data(),
                            reinterpret_cast<const int64_t *>(ids->data()), w->data(), ids->numel(),
                            w->shape()[1]);
    } else {
        LLAISYS_CUDA_LAUNCH((emb_k<2>), o->numel(), o->data(),
                            reinterpret_cast<const int64_t *>(ids->data()), w->data(), ids->numel(),
                            w->shape()[1]);
    }
}

} // namespace llaisys::ops::nvidia
