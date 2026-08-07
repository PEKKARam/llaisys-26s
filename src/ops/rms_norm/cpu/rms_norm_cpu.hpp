#pragma once
#include "llaisys.h"

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
} // namespace llaisys::ops::cpu