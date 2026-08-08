#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

namespace llaisys::ops::cpu {
template <typename T>
void rope_(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    const size_t S = out->shape()[0];
    const size_t H = out->shape()[1];
    const size_t D = out->shape()[2];
    const size_t half = D / 2;

    const T *input_data = reinterpret_cast<const T *>(in->data());
    const int64_t *positions = reinterpret_cast<const int64_t *>(pos_ids->data());
    T *output_data = reinterpret_cast<T *>(out->data());

    std::vector<float> freq_base(half);
    std::vector<float> sin_values(half);
    std::vector<float> cos_values(half);

    for (size_t j = 0; j < half; ++j) {
        const float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(D);
        freq_base[j] = std::pow(theta, exponent);
    }

    for (size_t seq = 0; seq < S; ++seq) {
        const float pos = static_cast<float>(positions[seq]);

        for (size_t j = 0; j < half; ++j) {
            const float angle = pos / freq_base[j];

            sin_values[j] = std::sin(angle);
            cos_values[j] = std::cos(angle);
        }

        for (size_t head = 0; head < H; ++head) {
            size_t base = (seq * H + head) * D;

            for (size_t j = 0; j < half; ++j) {
                float a = llaisys::utils::cast<float>(input_data[base + j]);
                float b = llaisys::utils::cast<float>(input_data[base + half + j]);

                float rotated_a = a * cos_values[j] - b * sin_values[j];
                float rotated_b = b * cos_values[j] + a * sin_values[j];

                output_data[base + j] = llaisys::utils::cast<T>(rotated_a);
                output_data[base + half + j] = llaisys::utils::cast<T>(rotated_b);
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(out, in, pos_ids, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_<llaisys::bf16_t>(out, in, pos_ids, theta);
    case LLAISYS_DTYPE_F16:
        return rope_<llaisys::fp16_t>(out, in, pos_ids, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}
} // namespace llaisys::ops::cpu
