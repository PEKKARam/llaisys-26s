#include "self_atten_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace llaisys::ops::cpu {
template <typename T>
void self_atten_(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    const size_t query_len = q->shape()[0];
    const size_t kv_len = k->shape()[0];
    const size_t query_heads = q->shape()[1];
    const size_t kv_heads = k->shape()[1];
    const size_t query_dim = q->shape()[2];
    const size_t value_dim = v->shape()[2];
    const size_t query_heads_per_kv_head = query_heads / kv_heads;
    const size_t cached_len = kv_len - query_len;

    const T *q_data = reinterpret_cast<const T *>(q->data());
    const T *k_data = reinterpret_cast<const T *>(k->data());
    const T *v_data = reinterpret_cast<const T *>(v->data());
    T *out_data = reinterpret_cast<T *>(attn_val->data());

    const float *q_values;
    const float *k_values;
    const float *v_values;
    std::vector<float> converted_q;
    std::vector<float> converted_k;
    std::vector<float> converted_v;

    if constexpr (std::is_same_v<T, float>) {
        q_values = q_data;
        k_values = k_data;
        v_values = v_data;
    } else {
        converted_q.resize(q->numel());
        converted_k.resize(k->numel());
        converted_v.resize(v->numel());
        std::transform(q_data, q_data + q->numel(), converted_q.begin(), [](T value) {
            return llaisys::utils::cast<float>(value);
        });
        std::transform(k_data, k_data + k->numel(), converted_k.begin(), [](T value) {
            return llaisys::utils::cast<float>(value);
        });
        std::transform(v_data, v_data + v->numel(), converted_v.begin(), [](T value) {
            return llaisys::utils::cast<float>(value);
        });
        q_values = converted_q.data();
        k_values = converted_k.data();
        v_values = converted_v.data();
    }

    std::vector<float> scores(kv_len);
    for (size_t query_head = 0; query_head < query_heads; ++query_head) {
        const size_t kv_head = query_head / query_heads_per_kv_head;

        for (size_t query_pos = 0; query_pos < query_len; ++query_pos) {
            // Queries describe the suffix of the KV cache, so each query can
            // attend through its matching absolute position in that cache.
            const size_t attended_len = cached_len + query_pos + 1;
            const size_t q_offset = (query_pos * query_heads + query_head) * query_dim;
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t key_pos = 0; key_pos < attended_len; ++key_pos) {
                const size_t k_offset = (key_pos * kv_heads + kv_head) * query_dim;
                float dot = 0.0f;
                for (size_t dim = 0; dim < query_dim; ++dim) {
                    dot += q_values[q_offset + dim] * k_values[k_offset + dim];
                }
                scores[key_pos] = dot * scale;
                max_score = std::max(max_score, scores[key_pos]);
            }

            float denominator = 0.0f;
            for (size_t key_pos = 0; key_pos < attended_len; ++key_pos) {
                scores[key_pos] = std::exp(scores[key_pos] - max_score);
                denominator += scores[key_pos];
            }

            const size_t out_offset = (query_pos * query_heads + query_head) * value_dim;
            for (size_t dim = 0; dim < value_dim; ++dim) {
                float result = 0.0f;
                for (size_t key_pos = 0; key_pos < attended_len; ++key_pos) {
                    const size_t v_offset = (key_pos * kv_heads + kv_head) * value_dim;
                    result += (scores[key_pos] / denominator) * v_values[v_offset + dim];
                }
                out_data[out_offset + dim] = llaisys::utils::cast<T>(result);
            }
        }
    }
}

void self_atten(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_atten_<float>(attn_val, q, k, v, scale);
    case LLAISYS_DTYPE_BF16:
        return self_atten_<llaisys::bf16_t>(attn_val, q, k, v, scale);
    case LLAISYS_DTYPE_F16:
        return self_atten_<llaisys::fp16_t>(attn_val, q, k, v, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}
} // namespace llaisys::ops::cpu
