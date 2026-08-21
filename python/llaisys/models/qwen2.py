from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import safetensors
import torch

from ..libllaisys import DataType
from ..libllaisys import DeviceType
from ..libllaisys import MemcpyKind
from ..libllaisys import LIB_LLAISYS
from ..runtime import RuntimeAPI
from ..tensor import Tensor
from ..ops import Ops


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        self.device_id = 0
        self.runtime = RuntimeAPI(self.device)

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

        self.in_embed: Tensor | None = None
        self.out_embed: Tensor | None = None
        self.out_norm_w: Tensor | None = None
        self.attn_norm_w: list[Tensor | None] = []
        self.attn_q_w: list[Tensor | None] = []
        self.attn_q_b: list[Tensor | None] = []
        self.attn_k_w: list[Tensor | None] = []
        self.attn_k_b: list[Tensor | None] = []
        self.attn_v_w: list[Tensor | None] = []
        self.attn_v_b: list[Tensor | None] = []
        self.attn_o_w: list[Tensor | None] = []
        self.mlp_norm_w: list[Tensor | None] = []
        self.mlp_gate_w: list[Tensor | None] = []
        self.mlp_up_w: list[Tensor | None] = []
        self.mlp_down_w: list[Tensor | None] = []

        self._cache_capacity = 0
        self._k_cache: list[Tensor] = []
        self._v_cache: list[Tensor] = []
        self._cached_len = 0

        self._load_weights()

    def _tensor_numel(self, shape: Sequence[int]) -> int:
        return math.prod(int(dim) for dim in shape)

    def _dtype_size(self, dtype: DataType) -> int:
        return {
            DataType.I64: 8,
            DataType.BF16: 2,
        }[dtype]

    def _make_tensor(self, shape: Sequence[int], dtype: DataType) -> Tensor:
        return Tensor(shape=shape, dtype=dtype, device=self.device, device_id=self.device_id)

    def _load_from_torch(self, dst: Tensor, src: torch.Tensor) -> None:
        if not src.is_contiguous():
            src = src.contiguous()
        dst.load(src.data_ptr())

    def _copy_tensor(self, dst: Tensor, src: Tensor) -> None:
        bytes_ = self._tensor_numel(dst.shape()) * self._dtype_size(dst.dtype())
        self.runtime.memcpy_sync(dst.data_ptr(), src.data_ptr(), bytes_, MemcpyKind.D2D)

    def _load_named_tensor(self, name: str, source: torch.Tensor, expected_shape: Sequence[int]) -> Tensor:
        if tuple(int(dim) for dim in source.shape) != tuple(int(dim) for dim in expected_shape):
            raise ValueError(f"{name} shape mismatch: expected {tuple(expected_shape)}, got {tuple(source.shape)}")
        if source.dtype != torch.bfloat16:
            raise ValueError(f"{name} dtype mismatch: expected torch.bfloat16, got {source.dtype}")
        tensor = self._make_tensor(expected_shape, self.dtype)
        self._load_from_torch(tensor, source)
        return tensor

    def _load_weights(self) -> None:
        for file in sorted(self.model_path.glob("*.safetensors")):
            with safetensors.safe_open(str(file), framework="pt", device="cpu") as file_data:
                for name in file_data.keys():
                    source = file_data.get_tensor(name)
                    self._assign_weight(name, source)

        if self.in_embed is None or self.out_norm_w is None:
            raise ValueError("Missing required Qwen2 weights")
        if self.tie_word_embeddings:
            self.out_embed = self.in_embed
        elif self.out_embed is None:
            raise ValueError("Missing lm_head.weight for untied embeddings")

        expected_layer_count = self.num_layers
        layer_lists = [
            self.attn_norm_w,
            self.attn_q_w,
            self.attn_q_b,
            self.attn_k_w,
            self.attn_k_b,
            self.attn_v_w,
            self.attn_v_b,
            self.attn_o_w,
            self.mlp_norm_w,
            self.mlp_gate_w,
            self.mlp_up_w,
            self.mlp_down_w,
        ]
        for weights in layer_lists:
            if len(weights) != expected_layer_count:
                raise ValueError("Incomplete layer weights for Qwen2")

    def _assign_weight(self, name: str, source: torch.Tensor) -> None:
        if name == "model.embed_tokens.weight":
            self.in_embed = self._load_named_tensor(name, source, source.shape)
            return
        if name == "lm_head.weight":
            self.out_embed = self._load_named_tensor(name, source, source.shape)
            return
        if name == "model.norm.weight":
            self.out_norm_w = self._load_named_tensor(name, source, source.shape)
            return

        if not name.startswith("model.layers."):
            raise ValueError(f"Unexpected Qwen2 weight: {name}")

        prefix, suffix = name.split(".", 3)[2], name.split(".", 3)[3]
        layer_index = int(prefix)

        while len(self.attn_norm_w) <= layer_index:
            self.attn_norm_w.append(None)
            self.attn_q_w.append(None)
            self.attn_q_b.append(None)
            self.attn_k_w.append(None)
            self.attn_k_b.append(None)
            self.attn_v_w.append(None)
            self.attn_v_b.append(None)
            self.attn_o_w.append(None)
            self.mlp_norm_w.append(None)
            self.mlp_gate_w.append(None)
            self.mlp_up_w.append(None)
            self.mlp_down_w.append(None)

        if suffix == "input_layernorm.weight":
            self.attn_norm_w[layer_index] = self._load_named_tensor(name, source, source.shape)
        elif suffix == "self_attn.q_proj.weight":
            self.attn_q_w[layer_index] = self._load_named_tensor(name, source, source.shape)
        elif suffix == "self_attn.q_proj.bias":
            self.attn_q_b[layer_index] = self._load_named_tensor(name, source, source.shape)
        elif suffix == "self_attn.k_proj.weight":
            self.attn_k_w[layer_index] = self._load_named_tensor(name, source, source.shape)
        elif suffix == "self_attn.k_proj.bias":
            self.attn_k_b[layer_index] = self._load_named_tensor(name, source, source.shape)
        elif suffix == "self_attn.v_proj.weight":
            self.attn_v_w[layer_index] = self._load_named_tensor(name, source, source.shape)
        elif suffix == "self_attn.v_proj.bias":
            self.attn_v_b[layer_index] = self._load_named_tensor(name, source, source.shape)
        elif suffix == "self_attn.o_proj.weight":
            self.attn_o_w[layer_index] = self._load_named_tensor(name, source, source.shape)
        elif suffix == "post_attention_layernorm.weight":
            self.mlp_norm_w[layer_index] = self._load_named_tensor(name, source, source.shape)
        elif suffix == "mlp.gate_proj.weight":
            self.mlp_gate_w[layer_index] = self._load_named_tensor(name, source, source.shape)
        elif suffix == "mlp.up_proj.weight":
            self.mlp_up_w[layer_index] = self._load_named_tensor(name, source, source.shape)
        elif suffix == "mlp.down_proj.weight":
            self.mlp_down_w[layer_index] = self._load_named_tensor(name, source, source.shape)
        else:
            raise ValueError(f"Unexpected Qwen2 weight: {name}")

    def _ensure_cache(self, capacity: int) -> None:
        if capacity <= self._cache_capacity:
            self._cached_len = 0
            return

        cache_shape = (capacity, self.num_kv_heads, self.head_dim)
        self._k_cache = [self._make_tensor(cache_shape, self.dtype) for _ in range(self.num_layers)]
        self._v_cache = [self._make_tensor(cache_shape, self.dtype) for _ in range(self.num_layers)]
        self._cache_capacity = capacity
        self._cached_len = 0

    def _make_ids_tensor(self, ids: Sequence[int]) -> Tensor:
        ids_tensor = self._make_tensor((len(ids),), DataType.I64)
        torch_ids = torch.tensor([int(token) for token in ids], dtype=torch.int64)
        self._load_from_torch(ids_tensor, torch_ids)
        return ids_tensor

    def _make_positions_tensor(self, start: int, end: int) -> Tensor:
        positions = self._make_tensor((end - start,), DataType.I64)
        torch_positions = torch.arange(start, end, dtype=torch.int64)
        self._load_from_torch(positions, torch_positions)
        return positions

    def _read_i64_scalar(self, tensor: Tensor) -> int:
        if self.device == DeviceType.CPU:
            raw_ptr = tensor.data_ptr()
            ptr = raw_ptr if isinstance(raw_ptr, int) else raw_ptr.value
            if ptr is None:
                raise RuntimeError("Tensor data pointer is null")
            return int(ctypes.c_int64.from_address(ptr).value)

        host_value = ctypes.c_int64()
        self.runtime.memcpy_sync(
            ctypes.byref(host_value),
            tensor.data_ptr(),
            ctypes.sizeof(host_value),
            MemcpyKind.D2H,
        )
        return int(host_value.value)

    def _require_weight(self, tensor: Tensor | None, name: str) -> Tensor:
        if tensor is None:
            raise RuntimeError(f"Missing Qwen2 weight: {name}")
        return tensor

    def _layer_forward(self, hidden: Tensor, layer_index: int, chunk_len: int) -> Tensor:
        residual = hidden

        attn_norm_w = self._require_weight(self.attn_norm_w[layer_index], f"attn_norm_w[{layer_index}]")
        attn_q_w = self._require_weight(self.attn_q_w[layer_index], f"attn_q_w[{layer_index}]")
        attn_q_b = self.attn_q_b[layer_index]
        attn_k_w = self._require_weight(self.attn_k_w[layer_index], f"attn_k_w[{layer_index}]")
        attn_k_b = self.attn_k_b[layer_index]
        attn_v_w = self._require_weight(self.attn_v_w[layer_index], f"attn_v_w[{layer_index}]")
        attn_v_b = self.attn_v_b[layer_index]
        attn_o_w = self._require_weight(self.attn_o_w[layer_index], f"attn_o_w[{layer_index}]")
        mlp_norm_w = self._require_weight(self.mlp_norm_w[layer_index], f"mlp_norm_w[{layer_index}]")
        mlp_gate_w = self._require_weight(self.mlp_gate_w[layer_index], f"mlp_gate_w[{layer_index}]")
        mlp_up_w = self._require_weight(self.mlp_up_w[layer_index], f"mlp_up_w[{layer_index}]")
        mlp_down_w = self._require_weight(self.mlp_down_w[layer_index], f"mlp_down_w[{layer_index}]")

        normed = self._make_tensor((chunk_len, self.hidden_size), self.dtype)
        Ops.rms_norm(normed, hidden, attn_norm_w, self.eps)

        q = self._make_tensor((chunk_len, self.hidden_size), self.dtype)
        Ops.linear(q, normed, attn_q_w, attn_q_b)
        q = q.view(chunk_len, self.num_heads, self.head_dim)

        kv_hidden = self.num_kv_heads * self.head_dim
        k = self._make_tensor((chunk_len, kv_hidden), self.dtype)
        Ops.linear(k, normed, attn_k_w, attn_k_b)
        k = k.view(chunk_len, self.num_kv_heads, self.head_dim)

        v = self._make_tensor((chunk_len, kv_hidden), self.dtype)
        Ops.linear(v, normed, attn_v_w, attn_v_b)
        v = v.view(chunk_len, self.num_kv_heads, self.head_dim)

        positions = self._make_positions_tensor(self._cached_len, self._cached_len + chunk_len)
        q_rot = self._make_tensor((chunk_len, self.num_heads, self.head_dim), self.dtype)
        Ops.rope(q_rot, q, positions, self.theta)
        k_rot = self._make_tensor((chunk_len, self.num_kv_heads, self.head_dim), self.dtype)
        Ops.rope(k_rot, k, positions, self.theta)

        self._copy_tensor(
            self._k_cache[layer_index].slice(0, self._cached_len, self._cached_len + chunk_len),
            k_rot,
        )
        self._copy_tensor(
            self._v_cache[layer_index].slice(0, self._cached_len, self._cached_len + chunk_len),
            v,
        )

        cached_end = self._cached_len + chunk_len
        attn_k = self._k_cache[layer_index].slice(0, 0, cached_end)
        attn_v = self._v_cache[layer_index].slice(0, 0, cached_end)

        attn_val = self._make_tensor((chunk_len, self.num_heads, self.head_dim), self.dtype)
        Ops.self_attention(attn_val, q_rot, attn_k, attn_v, 1.0 / math.sqrt(self.head_dim))

        attn_val = attn_val.view(chunk_len, self.hidden_size)
        attn_out = self._make_tensor((chunk_len, self.hidden_size), self.dtype)
        Ops.linear(attn_out, attn_val, attn_o_w, None)

        post_attn = self._make_tensor((chunk_len, self.hidden_size), self.dtype)
        Ops.add(post_attn, residual, attn_out)

        mlp_norm = self._make_tensor((chunk_len, self.hidden_size), self.dtype)
        Ops.rms_norm(mlp_norm, post_attn, mlp_norm_w, self.eps)

        gate = self._make_tensor((chunk_len, self.intermediate_size), self.dtype)
        Ops.linear(gate, mlp_norm, mlp_gate_w, None)

        up = self._make_tensor((chunk_len, self.intermediate_size), self.dtype)
        Ops.linear(up, mlp_norm, mlp_up_w, None)

        mlp_out = self._make_tensor((chunk_len, self.intermediate_size), self.dtype)
        Ops.swiglu(mlp_out, gate, up)

        down = self._make_tensor((chunk_len, self.hidden_size), self.dtype)
        Ops.linear(down, mlp_out, mlp_down_w, None)

        final_hidden = self._make_tensor((chunk_len, self.hidden_size), self.dtype)
        Ops.add(final_hidden, post_attn, down)
        return final_hidden

    def _forward_chunk(self, token_ids: Sequence[int]) -> int:
        chunk_len = len(token_ids)
        if chunk_len == 0:
            raise ValueError("token chunk must not be empty")

        in_embed = self._require_weight(self.in_embed, "in_embed")
        out_norm_w = self._require_weight(self.out_norm_w, "out_norm_w")
        out_embed = self._require_weight(self.out_embed, "out_embed")

        tokens = self._make_ids_tensor(token_ids)
        hidden = self._make_tensor((chunk_len, self.hidden_size), self.dtype)
        Ops.embedding(hidden, tokens, in_embed)

        for layer_index in range(self.num_layers):
            hidden = self._layer_forward(hidden, layer_index, chunk_len)

        final_hidden = self._make_tensor((chunk_len, self.hidden_size), self.dtype)
        Ops.rms_norm(final_hidden, hidden, out_norm_w, self.eps)

        logits = self._make_tensor((chunk_len, self.vocab_size), self.dtype)
        Ops.linear(logits, final_hidden, out_embed, None)

        last_logits = logits.slice(0, chunk_len - 1, chunk_len)
        max_idx = self._make_tensor((1, 1), DataType.I64)
        max_val = self._make_tensor((1, 1), self.dtype)
        Ops.argmax(max_idx, max_val, last_logits)
        return self._read_i64_scalar(max_idx)

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

        tokens = [int(token) for token in inputs]
        if not tokens:
            return []

        required_capacity = len(tokens) + max_new_tokens
        if required_capacity > self.max_position_embeddings:
            raise ValueError("Requested generation exceeds the model's maximum sequence length")

        self._ensure_cache(required_capacity)

        next_token = self._forward_chunk(tokens)
        tokens.append(next_token)
        self._cached_len += len(inputs)

        if next_token == self.end_token:
            return tokens

        for _ in range(max_new_tokens - 1):
            next_token = self._forward_chunk([tokens[-1]])
            self._cached_len += 1
            tokens.append(next_token)
            if next_token == self.end_token:
                break

        return tokens
