#include "../../nvidia/common.cuh"
#include "embedding_nvidia.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

// ids[n] selects rows from weight[vocab, d] into output[n, d].
template <int MODE>
__global__ void emb_k(void *o, const int64_t *ids, const void *w, size_t n,
                      size_t d) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * d) {
        store<MODE>(o, i, load<MODE>(w, (size_t)ids[i / d] * d + i % d));
    }
}

} // namespace

void embedding_v0(tensor_t o, tensor_t ids, tensor_t w) {
    int m = w->dtype() == LLAISYS_DTYPE_F32  ? 0
          : w->dtype() == LLAISYS_DTYPE_BF16 ? 1
                                             : 2;
    if (m == 0) {
        LLAISYS_CUDA_LAUNCH((emb_k<0>), o->numel(), o->data(),
                            reinterpret_cast<const int64_t *>(ids->data()),
                            w->data(), ids->numel(), w->shape()[1]);
    } else if (m == 1) {
        LLAISYS_CUDA_LAUNCH((emb_k<1>), o->numel(), o->data(),
                            reinterpret_cast<const int64_t *>(ids->data()),
                            w->data(), ids->numel(), w->shape()[1]);
    } else {
        LLAISYS_CUDA_LAUNCH((emb_k<2>), o->numel(), o->data(),
                            reinterpret_cast<const int64_t *>(ids->data()),
                            w->data(), ids->numel(), w->shape()[1]);
    }
}

namespace {
// One block per selected row, copying aligned 16-byte vectors without dtype
// conversion.
__global__ void embedding_vector_v2(void *out, const int64_t *ids,
                                    const void *weight, size_t vectors) {
    size_t row = blockIdx.x;
    const int4 *source
        = reinterpret_cast<const int4 *>(weight) + size_t(ids[row]) * vectors;
    int4 *target = reinterpret_cast<int4 *>(out) + row * vectors;
    for (size_t j = threadIdx.x; j < vectors; j += blockDim.x) {
        target[j] = source[j];
    }
}
} // namespace
void embedding(tensor_t o, tensor_t ids, tensor_t w) {
    static const int version = detail::selectedVersion("EMBEDDING");
    const size_t row_bytes = w->shape()[1] * w->elementSize();
    if (version < 2 || row_bytes % 16
        || reinterpret_cast<uintptr_t>(o->data()) % 16
        || reinterpret_cast<uintptr_t>(w->data()) % 16) {
        return embedding_v0(o, ids, w);
    }
    embedding_vector_v2<<<ids->numel(), 128, 0, detail::launchStream()>>>(
        o->data(), reinterpret_cast<const int64_t *>(ids->data()), w->data(),
        row_bytes / 16);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}
} // namespace llaisys::ops::nvidia
