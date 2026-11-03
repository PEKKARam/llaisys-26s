#include "../../nvidia/common.cuh"
#include "self_attention_nvidia.cuh"

#include "../../nvidia/resources.cuh"
#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

// Q[Q, HQ, D], K/V[KL, HK, D]; each KV head serves HQ/HK query heads.
// KL-Q is the cached prefix length, so query qp can see KL-Q+qp+1 keys.
template <int MODE>
__global__ void attn_k(void *o, const void *q, const void *k, const void *v,
                       size_t Q, size_t KL, size_t HQ, size_t HK, size_t D,
                       float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Q * HQ * D) {
        size_t d = i % D, h = (i / D) % HQ, qp = i / (HQ * D),
               kh = h / (HQ / HK), limit = KL - Q + qp + 1;
        float mx = -1e30f;
        for (size_t p = 0; p < limit; p++) {
            float s = 0;
            for (size_t z = 0; z < D; z++) {
                s += load<MODE>(q, (qp * HQ + h) * D + z)
                   * load<MODE>(k, (p * HK + kh) * D + z);
            }
            mx = fmaxf(mx, s * scale);
        }
        float den = 0, res = 0;
        for (size_t p = 0; p < limit; p++) {
            float s = 0;
            for (size_t z = 0; z < D; z++) {
                s += load<MODE>(q, (qp * HQ + h) * D + z)
                   * load<MODE>(k, (p * HK + kh) * D + z);
            }
            float e = expf(s * scale - mx);
            den += e;
            res += e * load<MODE>(v, (p * HK + kh) * D + d);
        }
        store<MODE>(o, i, res / den);
    }
}

} // namespace

void self_attention_v0(tensor_t o, tensor_t q, tensor_t k, tensor_t v,
                       float scale) {
    size_t Q = q->shape()[0], KL = k->shape()[0], HQ = q->shape()[1],
           HK = k->shape()[1], D = q->shape()[2];
    if (q->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((attn_k<0>), o->numel(), o->data(), q->data(),
                            k->data(), v->data(), Q, KL, HQ, HK, D, scale);
    } else if (q->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((attn_k<1>), o->numel(), o->data(), q->data(),
                            k->data(), v->data(), Q, KL, HQ, HK, D, scale);
    } else {
        LLAISYS_CUDA_LAUNCH((attn_k<2>), o->numel(), o->data(), q->data(),
                            k->data(), v->data(), Q, KL, HQ, HK, D, scale);
    }
}

namespace {
template <int MODE>
__device__ float score_v1(const void *q, const void *k, size_t qp, size_t h,
                          size_t kh, size_t p, size_t hq, size_t hk, size_t dim,
                          float scale) {
    float score = 0;
    for (size_t d = 0; d < dim; ++d) {
        score += load<MODE>(q, (qp * hq + h) * dim + d)
               * load<MODE>(k, (p * hk + kh) * dim + d);
    }
    return detail::rounded<MODE>(detail::rounded<MODE>(score) * scale);
}
template <int MODE>
__global__ void attn_serial_v1(void *out, const void *q, const void *k,
                               const void *v, size_t ql, size_t kl, size_t hq,
                               size_t hk, size_t dim, size_t dv, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ql * hq * dv) {
        return;
    }
    size_t d = i % dv, h = (i / dv) % hq, qp = i / (hq * dv),
           kh = h / (hq / hk), limit = kl - ql + qp + 1;
    float max_score = -INFINITY, den = 0, result = 0;
    for (size_t p = 0; p < limit; ++p) {
        max_score = fmaxf(
            max_score, score_v1<MODE>(q, k, qp, h, kh, p, hq, hk, dim, scale));
    }
    for (size_t p = 0; p < limit; ++p) {
        den += expf(score_v1<MODE>(q, k, qp, h, kh, p, hq, hk, dim, scale)
                    - max_score);
    }
    for (size_t p = 0; p < limit; ++p) {
        float prob = detail::rounded<MODE>(
            expf(score_v1<MODE>(q, k, qp, h, kh, p, hq, hk, dim, scale)
                 - max_score)
            / den);
        result += prob * load<MODE>(v, (p * hk + kh) * dv + d);
    }
    store<MODE>(out, i, result);
}

template <int MODE>
__global__ void qk_parallel_v2(float *scores, const void *q, const void *k,
                               size_t ql, size_t kl, size_t hq, size_t hk,
                               size_t dim, float scale) {
    size_t pair = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31;
    size_t row = pair / kl, p = pair % kl;
    if (row >= ql * hq) {
        return;
    }
    size_t qp = row / hq, h = row % hq, kh = h / (hq / hk);
    if (p >= kl - ql + qp + 1) {
        if (lane == 0) {
            scores[pair] = -INFINITY;
        }
        return;
    }
    float dot = 0;
    for (size_t d = lane; d < dim; d += 32) {
        dot += load<MODE>(q, (qp * hq + h) * dim + d)
             * load<MODE>(k, (p * hk + kh) * dim + d);
    }
    dot = detail::warpSum(dot);
    if (lane == 0) {
        scores[pair]
            = detail::rounded<MODE>(detail::rounded<MODE>(dot) * scale);
    }
}
template <int MODE>
__global__ void softmax_parallel_v2(float *scores, size_t kl) {
    size_t row = blockIdx.x;
    __shared__ float partial[8];
    float maximum = -INFINITY;
    for (size_t p = threadIdx.x; p < kl; p += blockDim.x) {
        maximum = fmaxf(maximum, scores[row * kl + p]);
    }
    maximum = detail::warpMax(maximum);
    if ((threadIdx.x & 31) == 0) {
        partial[threadIdx.x / 32] = maximum;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        maximum = threadIdx.x < 8 ? partial[threadIdx.x] : -INFINITY;
        maximum = detail::warpMax(maximum);
        if (threadIdx.x == 0) {
            partial[0] = maximum;
        }
    }
    __syncthreads();
    maximum = partial[0];
    float sum = 0;
    for (size_t p = threadIdx.x; p < kl; p += blockDim.x) {
        sum += expf(scores[row * kl + p] - maximum);
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
            partial[0] = sum;
        }
    }
    __syncthreads();
    sum = partial[0];
    for (size_t p = threadIdx.x; p < kl; p += blockDim.x) {
        scores[row * kl + p]
            = detail::rounded<MODE>(expf(scores[row * kl + p] - maximum) / sum);
    }
}
template <int MODE>
__global__ void pv_parallel_v2(void *out, const float *prob, const void *v,
                               size_t ql, size_t kl, size_t hq, size_t hk,
                               size_t dv) {
    size_t row = blockIdx.x, qp = row / hq, h = row % hq, kh = h / (hq / hk),
           limit = kl - ql + qp + 1;
    for (size_t d = threadIdx.x; d < dv; d += blockDim.x) {
        float sum = 0;
        for (size_t p = 0; p < limit; ++p) {
            sum += prob[row * kl + p] * load<MODE>(v, (p * hk + kh) * dv + d);
        }
        store<MODE>(out, row * dv + d, sum);
    }
}
template <int MODE>
void attn_run(tensor_t o, tensor_t q, tensor_t k, tensor_t v, float scale,
              int version) {
    if (version == 2) {
        size_t ql = q->shape()[0], kl = k->shape()[0], hq = q->shape()[1],
               hk = k->shape()[1], rows = ql * hq;
        float *scores = static_cast<float *>(
            detail::resources().attention.reserve(rows * kl * sizeof(float)));
        qk_parallel_v2<MODE>
            <<<(rows * kl + 7) / 8, 256, 0, detail::launchStream()>>>(
                scores, q->data(), k->data(), ql, kl, hq, hk, q->shape()[2],
                scale);
        detail::cudaCheck(cudaGetLastError());
        softmax_parallel_v2<MODE>
            <<<rows, 256, 0, detail::launchStream()>>>(scores, kl);
        detail::cudaCheck(cudaGetLastError());
        pv_parallel_v2<MODE><<<rows, 128, 0, detail::launchStream()>>>(
            o->data(), scores, v->data(), ql, kl, hq, hk, v->shape()[2]);
        detail::cudaCheck(cudaGetLastError());
        return;
    }
    LLAISYS_CUDA_LAUNCH((attn_serial_v1<MODE>), o->numel(), o->data(),
                        q->data(), k->data(), v->data(), q->shape()[0],
                        k->shape()[0], q->shape()[1], k->shape()[1],
                        q->shape()[2], v->shape()[2], scale);
}
} // namespace
void self_attention(tensor_t o, tensor_t q, tensor_t k, tensor_t v,
                    float scale) {
    static const int version = detail::selectedVersion("SELF_ATTENTION");
    if (version == 0) {
        return self_attention_v0(o, q, k, v, scale);
    }
    if (q->dtype() == LLAISYS_DTYPE_F32) {
        return attn_run<0>(o, q, k, v, scale, version);
    }
    if (q->dtype() == LLAISYS_DTYPE_BF16) {
        return attn_run<1>(o, q, k, v, scale, version);
    }
    return attn_run<2>(o, q, k, v, scale, version);
}
} // namespace llaisys::ops::nvidia
