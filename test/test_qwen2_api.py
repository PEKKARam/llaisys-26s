import ctypes

from test_utils import llaisys_device

import llaisys
from llaisys.libllaisys import LIB_LLAISYS, LlaisysQwen2Meta


def tensor_shape(handle):
    ndim = LIB_LLAISYS.tensorGetNdim(handle)
    shape = (ctypes.c_size_t * ndim)()
    LIB_LLAISYS.tensorGetShape(handle, shape)
    return tuple(shape)


def test_qwen2_model_lifecycle(device_name="cpu"):
    meta = LlaisysQwen2Meta(
        dtype=llaisys.DataType.BF16,
        nlayer=2,
        hs=8,
        nh=2,
        nkvh=1,
        dh=4,
        di=16,
        maxseq=32,
        voc=64,
        epsilon=1e-6,
        theta=10000.0,
        end_token=63,
    )
    device_ids = (ctypes.c_int * 1)(0)
    model = LIB_LLAISYS.llaisysQwen2ModelCreate(
        ctypes.byref(meta), llaisys_device(device_name), device_ids, 1
    )
    assert model

    try:
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(model)
        assert weights_ptr
        weights = weights_ptr.contents

        assert tensor_shape(weights.in_embed) == (64, 8)
        assert tensor_shape(weights.out_embed) == (64, 8)
        assert tensor_shape(weights.out_norm_w) == (8,)
        for layer in range(meta.nlayer):
            assert tensor_shape(weights.attn_norm_w[layer]) == (8,)
            assert tensor_shape(weights.attn_q_w[layer]) == (8, 8)
            assert tensor_shape(weights.attn_q_b[layer]) == (8,)
            assert tensor_shape(weights.attn_k_w[layer]) == (4, 8)
            assert tensor_shape(weights.attn_k_b[layer]) == (4,)
            assert tensor_shape(weights.attn_v_w[layer]) == (4, 8)
            assert tensor_shape(weights.attn_v_b[layer]) == (4,)
            assert tensor_shape(weights.attn_o_w[layer]) == (8, 8)
            assert tensor_shape(weights.mlp_norm_w[layer]) == (8,)
            assert tensor_shape(weights.mlp_gate_w[layer]) == (16, 8)
            assert tensor_shape(weights.mlp_up_w[layer]) == (16, 8)
            assert tensor_shape(weights.mlp_down_w[layer]) == (8, 16)

        LIB_LLAISYS.llaisysQwen2ModelReset(model)
    finally:
        LIB_LLAISYS.llaisysQwen2ModelDestroy(model)


if __name__ == "__main__":
    test_qwen2_model_lifecycle()
    print("\033[92mTest passed!\033[0m\n")
