#include "../../nvidia/common.cuh"
#include "linear_nvidia.cuh"

#include "../../nvidia/resources.cuh"
#include <climits>
#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

// X[M, K] * W[N, K]^T + bias[N] -> output[M, N].
// One thread computes one output element, accumulating across K in float.
template <int MODE>
__global__ void linear_k(void *o, const void *x, const void *w,
                         const void *bias, size_t M, size_t N, size_t K) {
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

void linear_v0(tensor_t o, tensor_t x, tensor_t w, tensor_t bias) {
    size_t K = x->shape().back(), M = x->numel() / K, N = w->shape()[0];
    if (w->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((linear_k<0>), o->numel(), o->data(), x->data(),
                            w->data(), bias ? bias->data() : nullptr, M, N, K);
    } else if (w->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((linear_k<1>), o->numel(), o->data(), x->data(),
                            w->data(), bias ? bias->data() : nullptr, M, N, K);
    } else {
        LLAISYS_CUDA_LAUNCH((linear_k<2>), o->numel(), o->data(), x->data(),
                            w->data(), bias ? bias->data() : nullptr, M, N, K);
    }
}

namespace {
template <int MODE>
__global__ void linear_bias(void *out, const float *product, const void *bias,
                            size_t count, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        store<MODE>(out, i, product[i] + load<MODE>(bias, i % n));
    }
}

template <int MODE>
void linear_blas(tensor_t out, tensor_t x, tensor_t w, tensor_t bias) {
    const size_t k = x->shape().back(), m = x->numel() / k, n = w->shape()[0];
    if (m > INT_MAX || n > INT_MAX || k > INT_MAX) {
        throw std::invalid_argument(
            "Linear dimensions exceed cuBLAS int range");
    }
    auto &state = detail::resources();
    detail::blasCheck(cublasSetStream(state.blas, detail::launchStream()));
    const cudaDataType_t dtype
        = MODE == 0 ? CUDA_R_32F : (MODE == 1 ? CUDA_R_16BF : CUDA_R_16F);
    const float alpha = 1.f, beta = 0.f;
    void *target = out->data();
    cudaDataType_t target_dtype = dtype;
    if (bias) {
        target = state.linear.reserve(m * n * sizeof(float));
        target_dtype = CUDA_R_32F;
    }
    // Row-major Y=XW^T is column-major Y^T=W X^T. No transpose copy is needed.
    detail::blasCheck(cublasGemmEx(
        state.blas, CUBLAS_OP_T, CUBLAS_OP_N, int(n), int(m), int(k), &alpha,
        w->data(), dtype, int(k), x->data(), dtype, int(k), &beta, target,
        target_dtype, int(n),
        MODE == 0 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
    if (bias) {
        LLAISYS_CUDA_LAUNCH((linear_bias<MODE>), m * n, out->data(),
                            static_cast<const float *>(target), bias->data(),
                            m * n, n);
    }
}
} // namespace
void linear(tensor_t out, tensor_t x, tensor_t w, tensor_t bias) {
    static const int version = detail::selectedVersion("LINEAR");
    if (version < 2) {
        return linear_v0(out, x, w, bias);
    }
    if (x->dtype() == LLAISYS_DTYPE_F32) {
        return linear_blas<0>(out, x, w, bias);
    }
    if (x->dtype() == LLAISYS_DTYPE_BF16) {
        return linear_blas<1>(out, x, w, bias);
    }
    return linear_blas<2>(out, x, w, bias);
}
} // namespace llaisys::ops::nvidia
