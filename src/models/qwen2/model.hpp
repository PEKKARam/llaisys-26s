#pragma once

#include "llaisys/models/qwen2.h"

#include "../../llaisys/llaisys_tensor.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaisys::models::qwen2 {

class Model {
private:
    using tensor_handle_t = std::unique_ptr<LlaisysTensor>;

    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;
    size_t _cached_len{0};
    int _engine_version{2};
    std::unordered_map<std::string, tensor_t> _workspace;
    size_t _workspace_allocations{0};
    std::vector<int64_t> _host_positions;

    LlaisysQwen2Weights _weights{};
    std::vector<tensor_handle_t> _owned_tensors;

    std::vector<llaisysTensor_t> _attn_norm_w;
    std::vector<llaisysTensor_t> _attn_q_w;
    std::vector<llaisysTensor_t> _attn_q_b;
    std::vector<llaisysTensor_t> _attn_k_w;
    std::vector<llaisysTensor_t> _attn_k_b;
    std::vector<llaisysTensor_t> _attn_v_w;
    std::vector<llaisysTensor_t> _attn_v_b;
    std::vector<llaisysTensor_t> _attn_o_w;
    std::vector<llaisysTensor_t> _mlp_norm_w;
    std::vector<llaisysTensor_t> _mlp_gate_w;
    std::vector<llaisysTensor_t> _mlp_up_w;
    std::vector<llaisysTensor_t> _mlp_down_w;

    // Per-layer KV cache.  The cache lives on the same device as the model
    // and is addressed as [maxseq, num_kv_heads, head_dim].
    std::vector<llaisys::tensor_t> _k_cache;
    std::vector<llaisys::tensor_t> _v_cache;

    llaisysTensor_t createTensor(const std::vector<size_t> &shape);
    void allocateWeights();
    void allocateCache();
    void exposeWeightArrays();
    tensor_t temporary(const char *name, const std::vector<size_t> &shape,
                       llaisysDataType_t dtype);

public:
    Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device,
          const int *device_ids, int ndevice);

    LlaisysQwen2Weights *weights();
    void reset();
    LlaisysQwen2State state() const;
    int64_t infer(const int64_t *token_ids, size_t ntoken);
};

} // namespace llaisys::models::qwen2
