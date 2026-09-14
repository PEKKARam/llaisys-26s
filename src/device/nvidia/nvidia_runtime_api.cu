#include "../runtime_api.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

namespace llaisys::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e == cudaErrorNoDevice) return 0;
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    return n;
}

void setDevice(int id) {
    cudaError_t e = cudaSetDevice(id);
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

void deviceSynchronize() {
    cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

llaisysStream_t createStream() {
    cudaStream_t s=nullptr; cudaError_t e=cudaStreamCreate(&s); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); return reinterpret_cast<llaisysStream_t>(s);
}

void destroyStream(llaisysStream_t stream) {
    if (stream) cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
}
void streamSynchronize(llaisysStream_t stream) {
    cudaError_t e=cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

void *mallocDevice(size_t size) {
    void *p=nullptr; cudaError_t e=cudaMalloc(&p,size); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); return p;
}

void freeDevice(void *ptr) {
    if(ptr) cudaFree(ptr);
}

void *mallocHost(size_t size) {
    void *p=nullptr; cudaError_t e=cudaMallocHost(&p,size); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); return p;
}

void freeHost(void *ptr) {
    if(ptr) cudaFreeHost(ptr);
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaMemcpyKind k = kind==LLAISYS_MEMCPY_H2D ? cudaMemcpyHostToDevice : kind==LLAISYS_MEMCPY_D2H ? cudaMemcpyDeviceToHost : kind==LLAISYS_MEMCPY_D2D ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToHost;
    cudaError_t e=cudaMemcpy(dst,src,size,k); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    cudaMemcpyKind k = kind==LLAISYS_MEMCPY_H2D ? cudaMemcpyHostToDevice : kind==LLAISYS_MEMCPY_D2H ? cudaMemcpyDeviceToHost : kind==LLAISYS_MEMCPY_D2D ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToHost;
    cudaError_t e=cudaMemcpyAsync(dst,src,size,k,reinterpret_cast<cudaStream_t>(stream)); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
