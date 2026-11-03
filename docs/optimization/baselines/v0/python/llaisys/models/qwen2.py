from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Optional, Sequence

import safetensors
import torch

from ..libllaisys import DataType
from ..libllaisys import DeviceType
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import LlaisysQwen2Meta


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        self.device_id = 0

        with open(self.model_path / "config.json", "r", encoding="utf-8") as file:
            config = json.load(file)

        self.model_type = config.get("model_type", "qwen2")
        self.hidden_size = int(config["hidden_size"])
        self.num_layers = int(config["num_hidden_layers"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = int(config["intermediate_size"])
        self.max_position_embeddings = int(config["max_position_embeddings"])
        self.vocab_size = int(config["vocab_size"])
        self.eps = float(config["rms_norm_eps"])
        self.theta = float(config.get("rope_theta", 10000.0))
        self.end_token = int(config.get("eos_token_id", -1))
        self.tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
        self.dtype = DataType.BF16

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        meta = LlaisysQwen2Meta(
            dtype=self.dtype, nlayer=self.num_layers, hs=self.hidden_size,
            nh=self.num_heads, nkvh=self.num_kv_heads, dh=self.head_dim,
            di=self.intermediate_size, maxseq=self.max_position_embeddings,
            voc=self.vocab_size, epsilon=self.eps, theta=self.theta,
            end_token=self.end_token,
        )
        device_ids = (ctypes.c_int * 1)(self.device_id)
        
        # 创建 C++ 模型，Python通过ctypes
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta), self.device, device_ids, 1
        )
        if not self._model:
            raise RuntimeError("llaisysQwen2ModelCreate returned null")
        self._weights_handle = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        if not self._weights_handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
            raise RuntimeError("llaisysQwen2ModelWeights returned null")

        self._load_weights()

    def __del__(self):
        model = getattr(self, "_model", None)
        if model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
            self._model = None

    def _load_weights(self) -> None:
        weights = self._weights_handle.contents
        loaded: set[str] = set()
        for file in sorted(self.model_path.glob("*.safetensors")):
            with safetensors.safe_open(str(file), framework="pt", device="cpu") as file_data:
                for name in file_data.keys():
                    source = file_data.get_tensor(name)
                    if name in loaded:
                        raise ValueError(f"Duplicate Qwen2 weight: {name}")
                    loaded.add(name)
                    self._assign_backend_weight(weights, name, source)

        required = {"model.embed_tokens.weight", "model.norm.weight"}
        for layer in range(self.num_layers):
            required.update({
                f"model.layers.{layer}.input_layernorm.weight",
                f"model.layers.{layer}.self_attn.q_proj.weight",
                f"model.layers.{layer}.self_attn.q_proj.bias",
                f"model.layers.{layer}.self_attn.k_proj.weight",
                f"model.layers.{layer}.self_attn.k_proj.bias",
                f"model.layers.{layer}.self_attn.v_proj.weight",
                f"model.layers.{layer}.self_attn.v_proj.bias",
                f"model.layers.{layer}.self_attn.o_proj.weight",
                f"model.layers.{layer}.post_attention_layernorm.weight",
                f"model.layers.{layer}.mlp.gate_proj.weight",
                f"model.layers.{layer}.mlp.up_proj.weight",
                f"model.layers.{layer}.mlp.down_proj.weight",
            })
        if not self.tie_word_embeddings:
            required.add("lm_head.weight")
        missing = sorted(required - loaded)
        if missing:
            raise ValueError(f"Missing Qwen2 weights: {missing[:4]}" + (" ..." if len(missing) > 4 else ""))

        if self.tie_word_embeddings and "lm_head.weight" not in loaded:
            source = self._read_weight_source("model.embed_tokens.weight")
            LIB_LLAISYS.tensorLoad(weights.out_embed, ctypes.c_void_p(source.data_ptr()))

    def _read_weight_source(self, wanted: str):
        for file in sorted(self.model_path.glob("*.safetensors")):
            with safetensors.safe_open(str(file), framework="pt", device="cpu") as data:
                if wanted in data.keys():
                    return data.get_tensor(wanted).contiguous()
        raise ValueError(f"Missing Qwen2 weight: {wanted}")

    def _assign_backend_weight(self, weights, name: str, source: torch.Tensor) -> None:
        if source.dtype != torch.bfloat16:
            raise ValueError(f"{name} dtype mismatch: expected torch.bfloat16, got {source.dtype}")
        source = source.contiguous()
        handle = None
        if name == "model.embed_tokens.weight":
            handle = weights.in_embed
        elif name == "lm_head.weight":
            handle = weights.out_embed
        elif name == "model.norm.weight":
            handle = weights.out_norm_w
        elif name.startswith("model.layers."):
            parts = name.split(".")
            layer = int(parts[2])
            suffix = ".".join(parts[3:])
            if layer >= self.num_layers:
                raise ValueError(f"Layer index out of range in {name}")
            mapping = {
                "input_layernorm.weight": weights.attn_norm_w[layer],
                "self_attn.q_proj.weight": weights.attn_q_w[layer],
                "self_attn.q_proj.bias": weights.attn_q_b[layer],
                "self_attn.k_proj.weight": weights.attn_k_w[layer],
                "self_attn.k_proj.bias": weights.attn_k_b[layer],
                "self_attn.v_proj.weight": weights.attn_v_w[layer],
                "self_attn.v_proj.bias": weights.attn_v_b[layer],
                "self_attn.o_proj.weight": weights.attn_o_w[layer],
                "post_attention_layernorm.weight": weights.mlp_norm_w[layer],
                "mlp.gate_proj.weight": weights.mlp_gate_w[layer],
                "mlp.up_proj.weight": weights.mlp_up_w[layer],
                "mlp.down_proj.weight": weights.mlp_down_w[layer],
            }
            handle = mapping.get(suffix)
        if handle is None:
            raise ValueError(f"Unexpected Qwen2 weight: {name}")
        ndim = int(LIB_LLAISYS.tensorGetNdim(handle))
        shape = (ctypes.c_size_t * ndim)()
        LIB_LLAISYS.tensorGetShape(handle, shape)
        expected = tuple(int(shape[i]) for i in range(ndim))
        if tuple(source.shape) != expected:
            raise ValueError(f"{name} shape mismatch: expected {expected}, got {tuple(source.shape)}")
        LIB_LLAISYS.tensorLoad(handle, ctypes.c_void_p(source.data_ptr()))

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if top_k != 1:
            raise NotImplementedError("Qwen2.generate currently supports argmax only (top_k == 1).")

        if max_new_tokens is None:
            max_new_tokens = 128
        
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        tokens = [int(token) for token in inputs]
        if not tokens:
            return []
        
        if max_new_tokens == 0:
            return tokens

        required_capacity = len(tokens) + max_new_tokens
        if required_capacity > self.max_position_embeddings:
            raise ValueError("Requested generation exceeds the model's maximum sequence length")

        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)
        ids = (ctypes.c_int64 * len(tokens))(*tokens)
        next_token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, ids, len(tokens)))
        tokens.append(next_token)

        if next_token == self.end_token:
            return tokens

        for _ in range(max_new_tokens - 1):
            ids = (ctypes.c_int64 * 1)(tokens[-1])
            next_token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, ids, 1))
            tokens.append(next_token)
            if next_token == self.end_token:
                break

        return tokens
