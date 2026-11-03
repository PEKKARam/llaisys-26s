#include "../../nvidia/common.cuh"
#include "self_attention_nvidia.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

// Q[Q, HQ, D], K/V[KL, HK, D]; each KV head serves HQ/HK query heads.
// KL-Q is the cached prefix length, so query qp can see KL-Q+qp+1 keys.
template <int MODE>
__global__ void attn_k(void *o, const void *q, const void *k, const void *v, size_t Q, size_t KL,
                       size_t HQ, size_t HK, size_t D, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Q * HQ * D) {
        size_t d = i % D, h = (i / D) % HQ, qp = i / (HQ * D), kh = h / (HQ / HK),
               limit = KL - Q + qp + 1;
        float mx = -1e30f;
        for (size_t p = 0; p < limit; p++) {
            float s = 0;
            for (size_t z = 0; z < D; z++) {
                s += load<MODE>(q, (qp * HQ + h) * D + z) * load<MODE>(k, (p * HK + kh) * D + z);
            }
            mx = fmaxf(mx, s * scale);
        }
        float den = 0, res = 0;
        for (size_t p = 0; p < limit; p++) {
            float s = 0;
            for (size_t z = 0; z < D; z++) {
                s += load<MODE>(q, (qp * HQ + h) * D + z) * load<MODE>(k, (p * HK + kh) * D + z);
            }
            float e = expf(s * scale - mx);
            den += e;
            res += e * load<MODE>(v, (p * HK + kh) * D + d);
        }
        store<MODE>(o, i, res / den);
    }
}

} // namespace

void self_attention(tensor_t o, tensor_t q, tensor_t k, tensor_t v, float scale) {
    size_t Q = q->shape()[0], KL = k->shape()[0], HQ = q->shape()[1], HK = k->shape()[1],
           D = q->shape()[2];
    if (q->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((attn_k<0>), o->numel(), o->data(), q->data(), k->data(), v->data(), Q,
                            KL, HQ, HK, D, scale);
    } else if (q->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((attn_k<1>), o->numel(), o->data(), q->data(), k->data(), v->data(), Q,
                            KL, HQ, HK, D, scale);
    } else {
        LLAISYS_CUDA_LAUNCH((attn_k<2>), o->numel(), o->data(), q->data(), k->data(), v->data(), Q,
                            KL, HQ, HK, D, scale);
    }
}

} // namespace llaisys::ops::nvidia
