#pragma once
#include "tensor.h"

__C {
    // Developer microbenchmark: private-stream CUDA Graph replay, device-event
    // milliseconds per call. Graph setup/lazy allocation are excluded.
    // Returns a negative value on invalid arguments or backend errors.
    __export float llaisysBenchmarkCudaOp(
        const char *name, llaisysTensor_t *tensors, size_t count, float scalar,
        int iterations);
}
