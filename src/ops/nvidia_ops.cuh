#pragma once

#include "../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void add(tensor_t, tensor_t, tensor_t);
void argmax(tensor_t, tensor_t, tensor_t);
void embedding(tensor_t, tensor_t, tensor_t);
void linear(tensor_t, tensor_t, tensor_t, tensor_t);
void rms_norm(tensor_t, tensor_t, tensor_t, float);
void rope(tensor_t, tensor_t, tensor_t, float);
void self_attention(tensor_t, tensor_t, tensor_t, tensor_t, float);
void swiglu(tensor_t, tensor_t, tensor_t);
}
