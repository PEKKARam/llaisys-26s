#include "../../nvidia/common.cuh"
#include "rope_nvidia.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

// Input/output: [S, H, D]. Pair the first and second halves of each head.
// pos[S] contains absolute positions, including the existing cache length.
template <int MODE>
__global__ void rope_k(void *o, const void *x, const int64_t *pos, size_t S,
                       size_t H, size_t D, float theta) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < S * H * D) {
        size_t d = i % D, j = d < (D / 2) ? d : d - D / 2;
        float a = load<MODE>(x, i);
        size_t base = i - d;
        size_t pair = base + (d < (D / 2) ? d + (D / 2) : d - (D / 2));
        float b = load<MODE>(x, pair);
        float ang = (float)pos[i / (H * D)] / powf(theta, 2.f * j / D);
        float v = d < (D / 2) ? a * cosf(ang) - b * sinf(ang)
                              : a * cosf(ang) + b * sinf(ang);
        store<MODE>(o, i, v);
    }
}

} // namespace

void rope_v0(tensor_t o, tensor_t x, tensor_t p, float th) {
    size_t S = x->shape()[0], H = x->shape()[1], D = x->shape()[2];
    if (x->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((rope_k<0>), o->numel(), o->data(), x->data(),
                            reinterpret_cast<const int64_t *>(p->data()), S, H,
                            D, th);
    } else if (x->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((rope_k<1>), o->numel(), o->data(), x->data(),
                            reinterpret_cast<const int64_t *>(p->data()), S, H,
                            D, th);
    } else {
        LLAISYS_CUDA_LAUNCH((rope_k<2>), o->numel(), o->data(), x->data(),
                            reinterpret_cast<const int64_t *>(p->data()), S, H,
                            D, th);
    }
}

namespace {
template <int MODE>
__global__ void rope_serial_v1(void *out, const void *x, const int64_t *pos,
                               size_t count, size_t heads, size_t dim,
                               float theta) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    size_t d = i % dim, j = d % (dim / 2), base = i - d;
    float angle
        = float(pos[i / (heads * dim)]) * (1.f / powf(theta, 2.f * j / dim));
    float c = detail::rounded<MODE>(cosf(angle)),
          s = detail::rounded<MODE>(sinf(angle));
    float a = load<MODE>(x, base + j), b = load<MODE>(x, base + j + dim / 2);
    float value
        = d < dim / 2
            ? detail::rounded<MODE>(a * c) - detail::rounded<MODE>(b * s)
            : detail::rounded<MODE>(b * c) + detail::rounded<MODE>(a * s);
    store<MODE>(out, i, value);
}

template <int MODE>
__global__ void rope_pair_v2(void *out, const void *x, const int64_t *pos,
                             size_t pairs, size_t heads, size_t dim,
                             float theta) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) {
        return;
    }
    size_t half = dim / 2, j = i % half, vector = i / half, base = vector * dim;
    float angle
        = float(pos[vector / heads]) * (1.f / powf(theta, 2.f * j / dim));
    float sine, cosine;
    sincosf(angle, &sine, &cosine);
    float c = detail::rounded<MODE>(cosine), sn = detail::rounded<MODE>(sine);
    float a = load<MODE>(x, base + j), b = load<MODE>(x, base + j + half);
    store<MODE>(out, base + j,
                detail::rounded<MODE>(a * c) - detail::rounded<MODE>(b * sn));
    store<MODE>(out, base + j + half,
                detail::rounded<MODE>(b * c) + detail::rounded<MODE>(a * sn));
}
template <int MODE>
void rope_run(tensor_t o, tensor_t x, tensor_t p, float theta, int version) {
    if (version == 2) {
        LLAISYS_CUDA_LAUNCH(
            (rope_pair_v2<MODE>), o->numel() / 2, o->data(), x->data(),
            reinterpret_cast<const int64_t *>(p->data()), o->numel() / 2,
            x->shape()[1], x->shape()[2], theta);
        return;
    }
    LLAISYS_CUDA_LAUNCH((rope_serial_v1<MODE>), o->numel(), o->data(),
                        x->data(), reinterpret_cast<const int64_t *>(p->data()),
                        o->numel(), x->shape()[1], x->shape()[2], theta);
}
} // namespace
void rope(tensor_t o, tensor_t x, tensor_t p, float theta) {
    static const int version = detail::selectedVersion("ROPE");
    if (version == 0) {
        return rope_v0(o, x, p, theta);
    }
    if (x->dtype() == LLAISYS_DTYPE_F32) {
        return rope_run<0>(o, x, p, theta, version);
    }
    if (x->dtype() == LLAISYS_DTYPE_BF16) {
        return rope_run<1>(o, x, p, theta, version);
    }
    return rope_run<2>(o, x, p, theta, version);
}
} // namespace llaisys::ops::nvidia
