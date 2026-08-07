#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

namespace llaisys::ops::cpu {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    const int64_t *indices = reinterpret_cast<const int64_t *>(index->data());
    const std::byte *weight_data = weight->data();
    std::byte *output_data = out->data();
    const size_t row_bytes = weight->shape()[1] * weight->elementSize();

    for (size_t position = 0; position < index->numel(); ++position) {
        const int64_t token = indices[position];
        CHECK_ARGUMENT(token >= 0 && token < static_cast<int64_t>(weight->shape()[0]),
                       "Embedding index out of range.");
        std::memcpy(output_data + position * row_bytes,
                    weight_data + static_cast<size_t>(token) * row_bytes,
                    row_bytes);
    }
}
} // namespace llaisys::ops::cpu
