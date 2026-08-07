#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <thread>
#include <type_traits>
#include <vector>

namespace llaisys::ops::cpu {
template <typename T>
void linear_(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    const size_t K = in->shape().back();
    const size_t M = in->numel() / K;
    const size_t N = weight->shape()[0];

    const T *input_data = reinterpret_cast<const T *>(in->data());
    const T *weight_data = reinterpret_cast<const T *>(weight->data());
    const T *bias_data = bias == nullptr ? nullptr : reinterpret_cast<const T *>(bias->data());
    T *output_data = reinterpret_cast<T *>(out->data());

    const float *input_values;
    const float *weight_values;
    const float *bias_values = nullptr;
    std::vector<float> converted_input;
    std::vector<float> converted_weight;
    std::vector<float> converted_bias;

    // data type transform once or not
    if constexpr (std::is_same_v<T, float>) {
        input_values = input_data;
        weight_values = weight_data;
        bias_values = bias_data;
    } else {
        // Convert once before the matrix multiplication. For F16/BF16 this
        // avoids repeating the same conversion for every output dot product.
        converted_input.resize(in->numel());
        converted_weight.resize(weight->numel());
        std::transform(input_data, input_data + in->numel(), converted_input.begin(), [](T value) {
            return llaisys::utils::cast<float>(value);
        });
        std::transform(weight_data, weight_data + weight->numel(), converted_weight.begin(), [](T value) {
            return llaisys::utils::cast<float>(value);
        });
        input_values = converted_input.data();
        weight_values = converted_weight.data();

        if (bias_data != nullptr) {
            converted_bias.resize(bias->numel());
            std::transform(bias_data, bias_data + bias->numel(), converted_bias.begin(), [](T value) {
                return llaisys::utils::cast<float>(value);
            });
            bias_values = converted_bias.data();
        }
    }

    auto compute_rows = [&](size_t begin, size_t end) {
        for (size_t m = begin; m < end; ++m) {
            const float *input_row = input_values + m * K;
            for (size_t n = 0; n < N; ++n) {
                const float *weight_row = weight_values + n * K;
                float accumulator = bias_values == nullptr ? 0.0f : bias_values[n];
                for (size_t k = 0; k < K; ++k) {
                    accumulator += input_row[k] * weight_row[k];
                }
                output_data[m * N + n] = llaisys::utils::cast<T>(accumulator);
            }
        }
    };

    // Output rows are independent, so split them across available CPU cores.
    const size_t hardware_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t thread_count = std::min(M, hardware_threads);
    const size_t rows_per_thread = (M + thread_count - 1) / thread_count;
    std::vector<std::thread> workers;
    workers.reserve(thread_count > 0 ? thread_count - 1 : 0);
    for (size_t thread = 1; thread < thread_count; ++thread) {
        const size_t begin = thread * rows_per_thread;
        const size_t end = std::min(M, begin + rows_per_thread);
        workers.emplace_back(compute_rows, begin, end);
    }
    compute_rows(0, std::min(M, rows_per_thread));
    for (auto &worker : workers) {
        worker.join();
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        return linear_<float>(out, in, weight, bias);
    case LLAISYS_DTYPE_BF16:
        return linear_<llaisys::bf16_t>(out, in, weight, bias);
    case LLAISYS_DTYPE_F16:
        return linear_<llaisys::fp16_t>(out, in, weight, bias);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}
} // namespace llaisys::ops::cpu
