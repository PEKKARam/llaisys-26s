#pragma once
#include "llaisys.h"

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void embedding(tensor_t out, tensor_t index, tensor_t weight);
} // namespace llaisys::ops::cpu