import ctypes

from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t


class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]


class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_o_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_gate_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_up_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_down_w", ctypes.POINTER(llaisysTensor_t)),
    ]


llaisysQwen2Model_t = ctypes.c_void_p

class LlaisysQwen2State(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in (
        "cached_len", "cache_capacity", "cache_bytes", "weight_bytes",
        "workspace_bytes", "workspace_allocations",
    )]


def load_qwen2(lib):
    lib.llaisysQwen2LastError.argtypes = []
    lib.llaisysQwen2LastError.restype = ctypes.c_char_p
    lib.llaisysQwen2ModelState.argtypes = [llaisysQwen2Model_t, ctypes.POINTER(LlaisysQwen2State)]
    lib.llaisysQwen2ModelState.restype = ctypes.c_int
    lib.llaisysQwen2ModelCreate.argtypes = [
        ctypes.POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t

    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelReset.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelReset.restype = None

    lib.llaisysQwen2ModelInfer.argtypes = [
        llaisysQwen2Model_t,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
    ]
    lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64
