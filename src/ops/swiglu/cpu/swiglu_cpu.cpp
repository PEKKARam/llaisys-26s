#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {
template <typename T>
void swiglu_(tensor_t out, tensor_t gate, tensor_t up) {
    const T *gate_data = reinterpret_cast<const T *>(gate->data());
    const T *up_data = reinterpret_cast<const T *>(up->data());
    T *out_data = reinterpret_cast<T *>(out->data());

    for (size_t index = 0; index < out->numel(); ++index) {
        const float gate_value = llaisys::utils::cast<float>(gate_data[index]);
        const float up_value = llaisys::utils::cast<float>(up_data[index]);
        const float sigmoid = gate_value >= 0.0f
                                ? 1.0f / (1.0f + std::exp(-gate_value))
                                : std::exp(gate_value) / (1.0f + std::exp(gate_value));
        out_data[index] = llaisys::utils::cast<T>(up_value * gate_value * sigmoid);
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return swiglu_<float>(out, gate, up);
    case LLAISYS_DTYPE_BF16:
        return swiglu_<llaisys::bf16_t>(out, gate, up);
    case LLAISYS_DTYPE_F16:
        return swiglu_<llaisys::fp16_t>(out, gate, up);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops::cpu
