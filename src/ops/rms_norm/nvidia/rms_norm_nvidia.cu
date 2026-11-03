#include "../../nvidia/common.cuh"
#include "rms_norm_nvidia.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

// One thread normalizes one row of X[M, K]; weight has K elements.
template <int MODE>
__global__ void rms_k(void *o, const void *x, const void *w, size_t M, size_t K,
                      float eps) {
    size_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M) {
        float s = 0;
        for (size_t k = 0; k < K; k++) {
            float v = load<MODE>(x, r * K + k);
            s += v * v;
        }
        float inv = rsqrtf(s / K + eps);
        for (size_t k = 0; k < K; k++) {
            store<MODE>(o, r * K + k,
                        load<MODE>(x, r * K + k) * load<MODE>(w, k) * inv);
        }
    }
}

} // namespace

void rms_norm_v0(tensor_t o, tensor_t x, tensor_t w, float e) {
    size_t K = x->shape().back(), M = x->numel() / K;
    if (x->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((rms_k<0>), M, o->data(), x->data(), w->data(), M,
                            K, e);
    } else if (x->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((rms_k<1>), M, o->data(), x->data(), w->data(), M,
                            K, e);
    } else {
        LLAISYS_CUDA_LAUNCH((rms_k<2>), M, o->data(), x->data(), w->data(), M,
                            K, e);
    }
}

namespace {
template <int MODE>
__global__ void rms_serial_v1(void *out, const void *x, const void *w,
                              size_t rows, size_t dim, float eps) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    float sum = 0;
    for (size_t j = 0; j < dim; ++j) {
        float v = load<MODE>(x, row * dim + j);
        sum += v * v;
    }
    float inv = rsqrtf(sum / dim + eps);
    for (size_t j = 0; j < dim; ++j) {
        float normalized
            = detail::rounded<MODE>(load<MODE>(x, row * dim + j) * inv);
        store<MODE>(out, row * dim + j, normalized * load<MODE>(w, j));
    }
}

template <int MODE>
__global__ void rms_parallel_v2(void *out, const void *x, const void *w,
                                size_t dim, float eps) {
    const size_t row = blockIdx.x;
    __shared__ float partial[8];
    float sum = 0;
    for (size_t j = threadIdx.x; j < dim; j += blockDim.x) {
        float v = load<MODE>(x, row * dim + j);
        sum += __fmul_rn(v, v);
    }
    sum = detail::warpSum(sum);
    if ((threadIdx.x & 31) == 0) {
        partial[threadIdx.x / 32] = sum;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < 8 ? partial[threadIdx.x] : 0;
        sum = detail::warpSum(sum);
        if (threadIdx.x == 0) {
            partial[0] = rsqrtf(sum / dim + eps);
        }
    }
    __syncthreads();
    float inv = partial[0];
    for (size_t j = threadIdx.x; j < dim; j += blockDim.x) {
        float normalized
            = detail::rounded<MODE>(load<MODE>(x, row * dim + j) * inv);
        store<MODE>(out, row * dim + j, normalized * load<MODE>(w, j));
    }
}
template <int MODE>
void rms_run(tensor_t o, tensor_t x, tensor_t w, float eps, int version) {
    size_t k = x->shape().back(), m = x->numel() / k;
    if (version == 2) {
        rms_parallel_v2<MODE><<<m, 256, 0, detail::launchStream()>>>(
            o->data(), x->data(), w->data(), k, eps);
        auto error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
        return;
    }
    LLAISYS_CUDA_LAUNCH((rms_serial_v1<MODE>), m, o->data(), x->data(),
                        w->data(), m, k, eps);
}
} // namespace
void rms_norm(tensor_t o, tensor_t x, tensor_t w, float eps) {
    static const int version = detail::selectedVersion("RMS_NORM");
    if (version == 0) {
        return rms_norm_v0(o, x, w, eps);
    }
    if (x->dtype() == LLAISYS_DTYPE_F32) {
        return rms_run<0>(o, x, w, eps, version);
    }
    if (x->dtype() == LLAISYS_DTYPE_BF16) {
        return rms_run<1>(o, x, w, eps, version);
    }
    return rms_run<2>(o, x, w, eps, version);
}
} // namespace llaisys::ops::nvidia
