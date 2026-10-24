#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void argmax(tensor_t idx, tensor_t val, tensor_t x);
} // namespace llaisys::ops::nvidia
