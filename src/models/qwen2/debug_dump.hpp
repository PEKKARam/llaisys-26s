#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::models::qwen2::debug {

// Lossless, contiguous host snapshot: raw bytes plus dtype/shape metadata.
void dumpTensor(const char *directory, const char *name, tensor_t tensor);

} // namespace llaisys::models::qwen2::debug
