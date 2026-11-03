#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia::detail {
// Normal inference uses the default stream. The developer benchmark temporarily
// redirects launches to a private stream so it can time a captured CUDA graph.
inline thread_local cudaStream_t active_stream = nullptr;
inline cudaStream_t launchStream() { return active_stream; }
inline int selectedVersion(const char *op) {
    const std::string key = std::string("LLAISYS_CUDA_") + op + "_VERSION";
    const char *value = std::getenv(key.c_str());
    if (!value) {
        value = std::getenv("LLAISYS_CUDA_VERSION");
    }
    if (!value) {
        return 2;
    }
    if (std::string(value) == "0") {
        return 0;
    }
    if (std::string(value) == "1") {
        return 1;
    }
    if (std::string(value) == "2") {
        return 2;
    }
    throw std::invalid_argument("CUDA version must be 0, 1, or 2");
}
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

// Make intermediate low-precision rounding explicit when matching HF eager.
template <int MODE> __device__ inline float rounded(float value) {
    if constexpr (MODE == 1) {
        return __bfloat162float(__float2bfloat16_rn(value));
    }
    if constexpr (MODE == 2) {
        return __half2float(__float2half_rn(value));
    }
    return value;
}

template <int MODE>
__device__ inline float2 loadPair(const void *pointer, size_t i) {
    if constexpr (MODE == 0) {
        return reinterpret_cast<const float2 *>(pointer)[i / 2];
    }
    const uint32_t bits = reinterpret_cast<const uint32_t *>(pointer)[i / 2];
    if constexpr (MODE == 1) {
        return make_float2(__bfloat162float(__ushort_as_bfloat16(bits & 65535)),
                           __bfloat162float(__ushort_as_bfloat16(bits >> 16)));
    }
    return make_float2(__half2float(__ushort_as_half(bits & 65535)),
                       __half2float(__ushort_as_half(bits >> 16)));
}
template <int MODE>
__device__ inline void storePair(void *pointer, size_t i, float2 value) {
    if constexpr (MODE == 0) {
        reinterpret_cast<float2 *>(pointer)[i / 2] = value;
    } else {
        uint32_t lo, hi;
        if constexpr (MODE == 1) {
            lo = __bfloat16_as_ushort(__float2bfloat16_rn(value.x));
            hi = __bfloat16_as_ushort(__float2bfloat16_rn(value.y));
        } else {
            lo = __half_as_ushort(__float2half_rn(value.x));
            hi = __half_as_ushort(__float2half_rn(value.y));
        }
        reinterpret_cast<uint32_t *>(pointer)[i / 2] = lo | (hi << 16);
    }
}
__device__ inline float warpSum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}
__device__ inline float warpMax(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

} // namespace llaisys::ops::nvidia::detail

// Preserve the existing default-stream launch and immediate error check.
#define LLAISYS_CUDA_LAUNCH(kernel, n, ...)                                    \
    do {                                                                       \
        kernel<<<((n) + 255) / 256, 256, 0,                                    \
                 ::llaisys::ops::nvidia::detail::launchStream()>>>(            \
            __VA_ARGS__);                                                      \
        cudaError_t e = cudaGetLastError();                                    \
        if (e != cudaSuccess)                                                  \
            throw std::runtime_error(cudaGetErrorString(e));                   \
    } while (0)
