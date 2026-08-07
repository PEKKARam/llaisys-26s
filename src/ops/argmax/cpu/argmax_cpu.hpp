#pragma once
#include "llaisys.h"

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
}