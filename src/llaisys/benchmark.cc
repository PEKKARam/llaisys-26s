#include "llaisys/benchmark.h"
#ifndef ENABLE_NVIDIA_API
extern "C" __export float
llaisysBenchmarkCudaOp(const char *, llaisysTensor_t *, size_t, float, int) {
    return -1.f;
}
#endif
