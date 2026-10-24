#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void rope(tensor_t out, tensor_t x, tensor_t positions, float theta);
} // namespace llaisys::ops::nvidia
