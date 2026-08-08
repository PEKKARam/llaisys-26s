#pragma once
#include "llaisys.h"

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void self_atten(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
} // namespace llaisys::ops::cpu