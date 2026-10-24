#include "model.hpp"

#include "../../utils/check.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../llaisys/llaisys_tensor.hpp"

#include <cmath>
#include <limits>
#include <cstring>

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
    allocateCache();
    exposeWeightArrays();
}

void Model::allocateCache() {
    const std::vector<size_t> shape{_meta.maxseq, _meta.nkvh, _meta.dh};
    _k_cache.reserve(_meta.nlayer);
    _v_cache.reserve(_meta.nlayer);
    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        _k_cache.push_back(Tensor::create(shape, _meta.dtype, _device, _device_id));
        _v_cache.push_back(Tensor::create(shape, _meta.dtype, _device, _device_id));
    }
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

    auto wrap = [](llaisysTensor_t handle) -> tensor_t {
        CHECK_ARGUMENT(handle != nullptr, "Qwen2 weight handle is null");
        return handle->tensor;
    };
    auto ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    ids->load(token_ids);
    auto hidden = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
    ops::embedding(hidden, ids, wrap(_weights.in_embed));

    const size_t old_len = _cached_len;
    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        auto residual = hidden;
        auto normed = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::rms_norm(normed, hidden, wrap(_weights.attn_norm_w[layer]), _meta.epsilon);

        const size_t qdim = _meta.nh * _meta.dh;
        const size_t kvdim = _meta.nkvh * _meta.dh;
        auto q_flat = Tensor::create({ntoken, qdim}, _meta.dtype, _device, _device_id);
        auto k_flat = Tensor::create({ntoken, kvdim}, _meta.dtype, _device, _device_id);
        auto v_flat = Tensor::create({ntoken, kvdim}, _meta.dtype, _device, _device_id);
        ops::linear(q_flat, normed, wrap(_weights.attn_q_w[layer]), wrap(_weights.attn_q_b[layer]));
        ops::linear(k_flat, normed, wrap(_weights.attn_k_w[layer]), wrap(_weights.attn_k_b[layer]));
        ops::linear(v_flat, normed, wrap(_weights.attn_v_w[layer]), wrap(_weights.attn_v_b[layer]));

        auto q = q_flat->view({ntoken, _meta.nh, _meta.dh});
        auto k = k_flat->view({ntoken, _meta.nkvh, _meta.dh});
        auto v = v_flat->view({ntoken, _meta.nkvh, _meta.dh});
        auto positions = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
        std::vector<int64_t> host_positions(ntoken);
        for (size_t i = 0; i < ntoken; ++i) host_positions[i] = static_cast<int64_t>(old_len + i);
        positions->load(host_positions.data());
        auto q_rot = Tensor::create({ntoken, _meta.nh, _meta.dh}, _meta.dtype, _device, _device_id);
        auto k_rot = Tensor::create({ntoken, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);
        ops::rope(q_rot, q, positions, _meta.theta);
        ops::rope(k_rot, k, positions, _meta.theta);

        const size_t cache_end = old_len + ntoken;
        auto k_dst = _k_cache[layer]->slice(0, old_len, cache_end);
        auto v_dst = _v_cache[layer]->slice(0, old_len, cache_end);
        const size_t bytes = ntoken * kvdim * _k_cache[layer]->elementSize();
        core::context().setDevice(_device, _device_id);
        core::context().runtime().api()->memcpy_sync(k_dst->data(), k_rot->data(), bytes, LLAISYS_MEMCPY_D2D);
        core::context().runtime().api()->memcpy_sync(v_dst->data(), v->data(), bytes, LLAISYS_MEMCPY_D2D);

        auto cached_k = _k_cache[layer]->slice(0, 0, cache_end);
        auto cached_v = _v_cache[layer]->slice(0, 0, cache_end);
        auto attn = Tensor::create({ntoken, _meta.nh, _meta.dh}, _meta.dtype, _device, _device_id);
        ops::self_attention(attn, q_rot, cached_k, cached_v, 1.0f / std::sqrt(static_cast<float>(_meta.dh)));
        auto attn_flat = attn->view({ntoken, _meta.hs});
        auto attn_out = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::linear(attn_out, attn_flat, wrap(_weights.attn_o_w[layer]), nullptr);
        auto post_attn = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::add(post_attn, residual, attn_out);

        auto mlp_norm = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::rms_norm(mlp_norm, post_attn, wrap(_weights.mlp_norm_w[layer]), _meta.epsilon);
        auto gate = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device, _device_id);
        auto up = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device, _device_id);
        ops::linear(gate, mlp_norm, wrap(_weights.mlp_gate_w[layer]), nullptr);
        ops::linear(up, mlp_norm, wrap(_weights.mlp_up_w[layer]), nullptr);
        auto mlp = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device, _device_id);
        ops::swiglu(mlp, gate, up);
        auto down = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::linear(down, mlp, wrap(_weights.mlp_down_w[layer]), nullptr);
        hidden = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::add(hidden, post_attn, down);
    }
    _cached_len += ntoken;

    auto final_norm = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
    ops::rms_norm(final_norm, hidden, wrap(_weights.out_norm_w), _meta.epsilon);
    auto logits = Tensor::create({ntoken, _meta.voc}, _meta.dtype, _device, _device_id);
    ops::linear(logits, final_norm, wrap(_weights.out_embed), nullptr);
    auto last_logits = logits->slice(0, ntoken - 1, ntoken);
    auto max_idx = Tensor::create({1, 1}, LLAISYS_DTYPE_I64, _device, _device_id);
    auto max_val = Tensor::create({1, 1}, _meta.dtype, _device, _device_id);
    ops::argmax(max_idx, max_val, last_logits);
    int64_t result = -1;
    core::context().setDevice(_device, _device_id);
    core::context().runtime().api()->memcpy_sync(&result, max_idx->data(), sizeof(result), LLAISYS_MEMCPY_D2H);
    return result;
}

} // namespace llaisys::models::qwen2
