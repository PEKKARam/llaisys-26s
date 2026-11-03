#include "../../llaisys/llaisys_tensor.hpp"
#include "../add/op.hpp"
#include "../argmax/op.hpp"
#include "../embedding/op.hpp"
#include "../linear/op.hpp"
#include "../rms_norm/op.hpp"
#include "../rope/op.hpp"
#include "../self_attention/op.hpp"
#include "../swiglu/op.hpp"
#include "common.cuh"
#include "llaisys/benchmark.h"
#include <functional>
#include <string>

extern "C" __export float llaisysBenchmarkCudaOp(const char *name,
                                                 llaisysTensor_t *handles,
                                                 size_t count, float scalar,
                                                 int iterations) {
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    auto previous = llaisys::ops::nvidia::detail::active_stream;
    try {
        if (!name || !handles || iterations < 1) {
            return -1.f;
        }
        std::string op(name);
        size_t expected = (op == "linear" || op == "self_attention") ? 4 : 3;
        if (count != expected) {
            return -1.f;
        }
        std::vector<llaisys::tensor_t> t;
        for (size_t i = 0; i < count; ++i) {
            t.push_back(handles[i] ? handles[i]->tensor : nullptr);
        }
        std::function<void()> call;
        if (op == "add") {
            call = [&] { llaisys::ops::add(t[0], t[1], t[2]); };
        } else if (op == "embedding") {
            call = [&] { llaisys::ops::embedding(t[0], t[1], t[2]); };
        } else if (op == "linear") {
            call = [&] { llaisys::ops::linear(t[0], t[1], t[2], t[3]); };
        } else if (op == "rms_norm") {
            call = [&] { llaisys::ops::rms_norm(t[0], t[1], t[2], scalar); };
        } else if (op == "rope") {
            call = [&] { llaisys::ops::rope(t[0], t[1], t[2], scalar); };
        } else if (op == "self_attention") {
            call = [&] {
                llaisys::ops::self_attention(t[0], t[1], t[2], t[3], scalar);
            };
        } else if (op == "swiglu") {
            call = [&] { llaisys::ops::swiglu(t[0], t[1], t[2]); };
        } else if (op == "argmax") {
            call = [&] { llaisys::ops::argmax(t[0], t[1], t[2]); };
        } else {
            return -1.f;
        }
        auto check = [](cudaError_t error) {
            if (error != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(error));
            }
        };
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        llaisys::ops::nvidia::detail::active_stream = stream;
        check(cudaDeviceSynchronize());
        call();
        call();
        check(cudaStreamSynchronize(
            stream)); // allocate lazy resources before capture
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < iterations; ++i) { call(); }
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, 0));
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        check(cudaEventCreate(&start));
        check(cudaEventCreate(&stop));
        check(cudaEventRecord(start, stream));
        check(cudaGraphLaunch(executable, stream));
        check(cudaEventRecord(stop, stream));
        check(cudaEventSynchronize(stop));
        float milliseconds = 0;
        check(cudaEventElapsedTime(&milliseconds, start, stop));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaGraphExecDestroy(executable);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        llaisys::ops::nvidia::detail::active_stream = previous;
        return milliseconds / iterations;
    } catch (const std::exception &error) {
        llaisys::ops::nvidia::detail::active_stream = previous;
        if (start) {
            cudaEventDestroy(start);
        }
        if (stop) {
            cudaEventDestroy(stop);
        }
        if (executable) {
            cudaGraphExecDestroy(executable);
        }
        if (graph) {
            cudaGraphDestroy(graph);
        }
        if (stream) {
            cudaStreamDestroy(stream);
        }
        std::cerr << "CUDA benchmark: " << error.what() << std::endl;
        return -1.f;
    }
}
