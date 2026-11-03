#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2Model;

    struct LlaisysQwen2State {
        size_t cached_len, cache_capacity, cache_bytes;
        size_t weight_bytes, workspace_bytes, workspace_allocations;
    };

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta, llaisysDeviceType_t device,
        const int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(
        struct LlaisysQwen2Model * model);

    __export void llaisysQwen2ModelReset(struct LlaisysQwen2Model * model);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model,
                                            const int64_t *token_ids,
                                            size_t ntoken);

    // Model APIs catch C++ exceptions. Infer returns -1 on error; other pointer
    // APIs return null. The error string is thread-local and cleared by the
    // next call.
    __export const char *llaisysQwen2LastError();
    __export int llaisysQwen2ModelState(struct LlaisysQwen2Model * model,
                                        struct LlaisysQwen2State * state);
}
#endif // LLAISYS_MODELS_QWEN2_H
