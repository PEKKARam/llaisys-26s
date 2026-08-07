#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>

namespace llaisys::ops::cpu {
template <typename T>
void rms_norm_(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    const T *in_data = reinterpret_cast<const T *>(in->data());
    const T *weight_data = reinterpret_cast<const T *>(weight->data());
    T *out_data = reinterpret_cast<T *>(out->data());

    const size_t K = in->shape().back();
    const size_t M = in->numel() / K;

    const float *in_values;
    const float *weight_values;
    std::vector<float> converted_input;
    std::vector<float> converted_weight;

    // data type transform once or not
    if constexpr (std::is_same_v<T, float>) {
        in_values = in_data;
        weight_values = weight_data;
    } else {
        converted_input.resize(in->numel());
        converted_weight.resize(weight->numel());
        std::transform(in_data, in_data + in->numel(), converted_input.begin(), [](T value) {
            return llaisys::utils::cast<float>(value);
        });
        std::transform(weight_data, weight_data + weight->numel(), converted_weight.begin(), [](T value) {
            return llaisys::utils::cast<float>(value);
        });
        in_values = converted_input.data();
        weight_values = converted_weight.data();
    }

    // 2D tensor
    for (size_t row = 0; row < M; ++row) {
        float sum_sq = 0.0f;
        size_t row_start = row * K;
        for (size_t idx = 0; idx < K; ++idx) {
            const float val = in_values[row_start + idx];
            sum_sq += val * val;
        }
        float mean_sq = sum_sq / llaisys::utils::cast<float>(K);

        const float inv_rms = 1.0f / sqrtf(mean_sq + eps);

        for (size_t idx = 0; idx < K; ++idx) {
            const size_t offset = row_start + idx;
            float result = weight_values[idx] * in_values[offset] * inv_rms;
            out_data[offset] = llaisys::utils::cast<T>(result);
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_<float>(out, in, weight, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_<llaisys::bf16_t>(out, in, weight, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_<llaisys::fp16_t>(out, in, weight, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}
} // namespace llaisys::ops::cpu