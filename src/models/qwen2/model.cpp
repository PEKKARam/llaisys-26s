#include "model.hpp"

#include "../../utils/check.hpp"

#include <limits>

namespace llaisys::models::qwen2 {
namespace {

void validateMeta(const LlaisysQwen2Meta &meta) {
    CHECK_ARGUMENT(meta.nlayer > 0, "Qwen2 must contain at least one decoder layer");
    CHECK_ARGUMENT(meta.hs > 0, "Qwen2 hidden size must be positive");
    CHECK_ARGUMENT(meta.nh > 0, "Qwen2 attention head count must be positive");
    CHECK_ARGUMENT(meta.nkvh > 0, "Qwen2 KV head count must be positive");
    CHECK_ARGUMENT(meta.dh > 0, "Qwen2 head dimension must be positive");
    CHECK_ARGUMENT(meta.di > 0, "Qwen2 intermediate size must be positive");
    CHECK_ARGUMENT(meta.maxseq > 0, "Qwen2 maximum sequence length must be positive");
    CHECK_ARGUMENT(meta.voc > 0, "Qwen2 vocabulary size must be positive");
    CHECK_ARGUMENT(meta.hs == meta.nh * meta.dh, "hidden size must equal attention heads times head dimension");
    CHECK_ARGUMENT(meta.nh % meta.nkvh == 0, "attention head count must be divisible by KV head count");
    CHECK_ARGUMENT(meta.epsilon > 0.0f, "RMSNorm epsilon must be positive");
    CHECK_ARGUMENT(meta.theta > 0.0f, "RoPE theta must be positive");
    CHECK_ARGUMENT(
        meta.dtype == LLAISYS_DTYPE_F32
            || meta.dtype == LLAISYS_DTYPE_F16
            || meta.dtype == LLAISYS_DTYPE_BF16,
        "Qwen2 weights must use F32, F16, or BF16");
}

} // namespace

Model::Model(
    const LlaisysQwen2Meta &meta,
    llaisysDeviceType_t device,
    const int *device_ids,
    int ndevice)
    : _meta(meta), _device(device), _device_id(0) {
    validateMeta(meta);
    CHECK_ARGUMENT(ndevice == 1, "the initial Qwen2 backend supports exactly one device");
    CHECK_ARGUMENT(device_ids != nullptr, "Qwen2 device_ids must not be null");
    CHECK_ARGUMENT(device_ids[0] >= 0, "Qwen2 device ID must not be negative");
    _device_id = device_ids[0];

    const size_t tensors_per_layer = 12;
    CHECK_ARGUMENT(
        meta.nlayer <= (std::numeric_limits<size_t>::max() - 3) / tensors_per_layer,
        "Qwen2 layer count is too large");
    _owned_tensors.reserve(3 + tensors_per_layer * meta.nlayer);
    allocateWeights();
    exposeWeightArrays();
}

llaisysTensor_t Model::createTensor(const std::vector<size_t> &shape) {
    auto tensor = std::make_unique<LlaisysTensor>();
    tensor->tensor = Tensor::create(shape, _meta.dtype, _device, _device_id);
    auto *handle = tensor.get();
    _owned_tensors.push_back(std::move(tensor));
    return handle;
}

void Model::allocateWeights() {
    const size_t query_size = _meta.nh * _meta.dh;
    const size_t kv_size = _meta.nkvh * _meta.dh;

    _weights.in_embed = createTensor({_meta.voc, _meta.hs});
    _weights.out_embed = createTensor({_meta.voc, _meta.hs});
    _weights.out_norm_w = createTensor({_meta.hs});

    auto allocate_per_layer = [this](auto &handles, const std::vector<size_t> &shape) {
        handles.reserve(_meta.nlayer);
        for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
            handles.push_back(createTensor(shape));
        }
    };

    allocate_per_layer(_attn_norm_w, {_meta.hs});
    allocate_per_layer(_attn_q_w, {query_size, _meta.hs});
    allocate_per_layer(_attn_q_b, {query_size});
    allocate_per_layer(_attn_k_w, {kv_size, _meta.hs});
    allocate_per_layer(_attn_k_b, {kv_size});
    allocate_per_layer(_attn_v_w, {kv_size, _meta.hs});
    allocate_per_layer(_attn_v_b, {kv_size});
    allocate_per_layer(_attn_o_w, {_meta.hs, query_size});
    allocate_per_layer(_mlp_norm_w, {_meta.hs});
    allocate_per_layer(_mlp_gate_w, {_meta.di, _meta.hs});
    allocate_per_layer(_mlp_up_w, {_meta.di, _meta.hs});
    allocate_per_layer(_mlp_down_w, {_meta.hs, _meta.di});
}

void Model::exposeWeightArrays() {
    _weights.attn_norm_w = _attn_norm_w.data();
    _weights.attn_q_w = _attn_q_w.data();
    _weights.attn_q_b = _attn_q_b.data();
    _weights.attn_k_w = _attn_k_w.data();
    _weights.attn_k_b = _attn_k_b.data();
    _weights.attn_v_w = _attn_v_w.data();
    _weights.attn_v_b = _attn_v_b.data();
    _weights.attn_o_w = _attn_o_w.data();
    _weights.mlp_norm_w = _mlp_norm_w.data();
    _weights.mlp_gate_w = _mlp_gate_w.data();
    _weights.mlp_up_w = _mlp_up_w.data();
    _weights.mlp_down_w = _mlp_down_w.data();
}

LlaisysQwen2Weights *Model::weights() {
    return &_weights;
}

void Model::reset() {
    _cached_len = 0;
}

int64_t Model::infer(const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(token_ids != nullptr, "Qwen2 token IDs must not be null");
    CHECK_ARGUMENT(ntoken > 0, "Qwen2 inference requires at least one token");
    CHECK_ARGUMENT(ntoken <= _meta.maxseq - _cached_len, "Qwen2 inference exceeds maximum sequence length");
    // The model lifecycle and weight ABI are implemented first. Returning an
    // invalid token keeps the unfinished forward path from throwing through C.
    return -1;
}

} // namespace llaisys::models::qwen2
