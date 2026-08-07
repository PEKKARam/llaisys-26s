#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <type_traits>

namespace llaisys::ops::cpu {

template <typename T>
void argmax_(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    const T *input = reinterpret_cast<const T *>(vals->data());
    T *output_val = reinterpret_cast<T *>(max_val->data());

    int64_t *output_idx = reinterpret_cast<int64_t *>(max_idx->data());

    const size_t last_dim = vals->shape().back();
    const size_t outer_numel = vals->numel() / last_dim; // outer dimensions numel except the last dimension

    for (size_t outer = 0; outer < outer_numel; ++outer) {
        const T *input_row = input + outer * last_dim;
        T curr_val = input_row[0];
        size_t curr_idx = 0;

        if constexpr (std::is_same_v<T, float>
                      || std::is_same_v<T, llaisys::fp16_t>
                      || std::is_same_v<T, llaisys::bf16_t>) {
            float curr_val_f32 = llaisys::utils::cast<float>(curr_val);
            for (size_t index = 1; index < last_dim; ++index) {
                const float candidate_f32 = llaisys::utils::cast<float>(input_row[index]);
                // Match torch.max: propagate the first NaN encountered.
                const bool is_better = std::isnan(candidate_f32)
                                           ? !std::isnan(curr_val_f32)
                                           : candidate_f32 > curr_val_f32;
                if (is_better) {
                    curr_val = input_row[index];
                    curr_val_f32 = candidate_f32;
                    curr_idx = index;
                }
            }
        } else {
            for (size_t index = 1; index < last_dim; ++index) {
                if (input_row[index] > curr_val) {
                    curr_val = input_row[index];
                    curr_idx = index;
                }
            }
        }

        output_val[outer] = curr_val;
        output_idx[outer] = static_cast<int64_t>(curr_idx);
    }
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // check data type and in-outputs type
    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
        return argmax_<float>(max_idx, max_val, vals);
    case LLAISYS_DTYPE_BF16:
        return argmax_<llaisys::bf16_t>(max_idx, max_val, vals);
    case LLAISYS_DTYPE_F16:
        return argmax_<llaisys::fp16_t>(max_idx, max_val, vals);
    case LLAISYS_DTYPE_I8:
        return argmax_<int8_t>(max_idx, max_val, vals);
    case LLAISYS_DTYPE_I16:
        return argmax_<int16_t>(max_idx, max_val, vals);
    case LLAISYS_DTYPE_I32:
        return argmax_<int32_t>(max_idx, max_val, vals);
    case LLAISYS_DTYPE_I64:
        return argmax_<int64_t>(max_idx, max_val, vals);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }
}
} // namespace llaisys::ops::cpu
