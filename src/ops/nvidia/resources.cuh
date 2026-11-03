#pragma once
#include "common.cuh"
#include <cublas_v2.h>
#include <memory>
#include <unordered_map>

namespace llaisys::ops::nvidia::detail {
inline void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}
inline void blasCheck(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS failure code "
                                 + std::to_string(int(status)));
    }
}
struct Scratch {
    void *pointer = nullptr;
    size_t capacity = 0;
    ~Scratch() {
        if (pointer) {
            cudaFree(pointer);
        }
    }
    void *reserve(size_t bytes) {
        if (bytes > capacity) {
            if (pointer) {
                cudaCheck(cudaFree(pointer));
            }
            pointer = nullptr;
            capacity = 0;
            cudaCheck(cudaMalloc(&pointer, bytes));
            capacity = bytes;
        }
        return pointer;
    }
};
struct Resources {
    int device;
    cublasHandle_t blas = nullptr;
    Scratch linear, attention, argmax;
    explicit Resources(int id) : device(id) {
        blasCheck(cublasCreate(&blas));
        blasCheck(cublasSetMathMode(
            blas, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
        blasCheck(cublasSetStream(blas, 0));
    }
    ~Resources() {
        int previous = device;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        if (blas) {
            cublasDestroy(blas);
        }
        if (linear.pointer) {
            cudaFree(linear.pointer);
        }
        if (attention.pointer) {
            cudaFree(attention.pointer);
        }
        if (argmax.pointer) {
            cudaFree(argmax.pointer);
        }
        linear.pointer = attention.pointer = argmax.pointer = nullptr;
        cudaSetDevice(previous);
    }
};
inline Resources &resources() {
    int device = 0;
    cudaCheck(cudaGetDevice(&device));
    thread_local std::unordered_map<int, std::unique_ptr<Resources>> states;
    auto &state = states[device];
    if (!state) {
        state = std::make_unique<Resources>(device);
    }
    return *state;
}
} // namespace llaisys::ops::nvidia::detail
