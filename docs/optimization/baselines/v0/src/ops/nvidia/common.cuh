#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace llaisys::ops::nvidia::detail {
// Storage modes: 0 = FP32, 1 = BF16, 2 = FP16. Arithmetic uses float.
__device__ inline float read16(const uint16_t *p, int mode) {
    if (mode == 1) {
        return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(p));
    }
    return __half2float(*reinterpret_cast<const __half *>(p));
}
__device__ inline void write16(uint16_t *p, float v, int mode) {
    if (mode == 1) {
        *reinterpret_cast<__nv_bfloat16 *>(p) = __float2bfloat16(v);
    } else {
        *reinterpret_cast<__half *>(p) = __float2half(v);
    }
}
template <int MODE> __device__ inline float load(const void *p, size_t i) {
    if constexpr (MODE != 0) {
        return read16(reinterpret_cast<const uint16_t *>(p) + i, MODE);
    }
    return reinterpret_cast<const float *>(p)[i];
}
template <int MODE> __device__ inline void store(void *p, size_t i, float v) {
    if constexpr (MODE != 0) {
        write16(reinterpret_cast<uint16_t *>(p) + i, v, MODE);
    } else {
        reinterpret_cast<float *>(p)[i] = v;
    }
}

} // namespace llaisys::ops::nvidia::detail

// Preserve the existing default-stream launch and immediate error check.
#define LLAISYS_CUDA_LAUNCH(kernel, n, ...)                                    \
    do {                                                                       \
        kernel<<<((n) + 255) / 256, 256>>>(__VA_ARGS__);                       \
        cudaError_t e = cudaGetLastError();                                    \
        if (e != cudaSuccess)                                                  \
            throw std::runtime_error(cudaGetErrorString(e));                   \
    } while (0)
