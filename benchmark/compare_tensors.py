"""Compare prefill checkpoints, optionally full layer 0 and final logits."""

import argparse
import gc
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from compare_prompt import configure_local_python_headers, configure_reference_math
from run_prompt import REPO_ROOT, load_sample


CHECKPOINTS = ("embedding", "layer0.input_norm", "layer0.output")
FULL_CHECKPOINTS = (
    "embedding",
    "layer0.input_norm",
    "layer0.q_proj",
    "layer0.k_proj",
    "layer0.v_proj",
    "layer0.q_rope",
    "layer0.k_rope",
    "layer0.attention",
    "layer0.post_attn",
    "layer0.mlp_norm",
    "layer0.gate",
    "layer0.up",
    "layer0.swiglu",
    "layer0.down",
    "layer0.output",
    "final.norm",
    "final.logits",
)


def read_native_tensor(directory, name):
    import torch

    meta_path = directory / f"{name}.json"
    data_path = directory / f"{name}.bin"
    if not meta_path.is_file() or not data_path.is_file():
        raise RuntimeError(
            f"缺少 {name} 的 native 导出，请先执行 xmake 和 xmake install"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["format"] != "llaisys_raw_tensor_v1" or meta["byte_order"] != sys.byteorder:
        raise ValueError(f"无法读取 {name} 的格式或字节序")
    dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[
        meta["dtype"]
    ]
    data = bytearray(data_path.read_bytes())
    expected = math.prod(meta["shape"]) * torch.empty((), dtype=dtype).element_size()
    if len(data) != expected or len(data) != meta["bytes"]:
        raise ValueError(f"{name} 的二进制大小与 shape/dtype 不一致")
    return torch.frombuffer(data, dtype=dtype).reshape(meta["shape"]).float().clone()


def tensor_errors(reference, actual):
    import torch

    if reference.shape != actual.shape:
        raise ValueError(
            f"shape 不同：HF={list(reference.shape)}, native={list(actual.shape)}"
        )
    finite = torch.isfinite(reference) & torch.isfinite(actual)
    same = reference == actual
    errors = (reference[finite].double() - actual[finite].double()).abs()
    different = (~same).nonzero()
    first = None
    if different.numel():
        index = tuple(different[0].tolist())
        first = {"index": list(index)}
        # Non-finite values are represented as strings to keep JSON valid.
        for key, tensor in (("hf_value", reference), ("native_value", actual)):
            value = float(tensor[index])
            first[key] = value if math.isfinite(value) else str(value)
    return {
        "shape": list(reference.shape),
        "elements": reference.numel(),
        "equal_elements": int(same.sum()),
        "equal_fraction": float(same.float().mean()),
        "values_equal": bool(same.all()),
        "hf_nonfinite": int((~torch.isfinite(reference)).sum()),
        "native_nonfinite": int((~torch.isfinite(actual)).sum()),
        "finite_pairs": int(finite.sum()),
        "max_abs_error": float(errors.max()) if errors.numel() else None,
        "mean_abs_error": float(errors.mean()) if errors.numel() else None,
        "rmse": float(errors.square().mean().sqrt()) if errors.numel() else None,
        "first_difference": first,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default="row_3_len_64")
    parser.add_argument("--full-layer", action="store_true")
    parser.add_argument(
        "--model", type=Path, default=REPO_ROOT / "DeepSeek-R1-Distill-Qwen-1.5B"
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=REPO_ROOT / "cache/prompt_sets/wikitext2_smoke.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "cache/tensor_debug",
        help="Parent directory; each run creates a fresh subdirectory",
    )
    args = parser.parse_args()
    checkpoints = FULL_CHECKPOINTS if args.full_layer else CHECKPOINTS
    sample = load_sample(args.prompts, args.sample_id)
    ids = sample["input_ids"]
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    if (
        sample["input_mode"] != "raw_completion"
        or not ids
        or len(ids) != sample["prompt_tokens"]
    ):
        parser.error("需要非空且长度匹配的 raw_completion 输入")
    if any(
        type(token) is not int or not 0 <= token < config["vocab_size"] for token in ids
    ):
        parser.error("输入包含非法 token ID")
    if len(ids) + 1 > config["max_position_embeddings"]:
        parser.error("输入超过生成接口的序列长度上限")

    configure_local_python_headers()
    import torch
    import transformers
    import llaisys
    from transformers import AutoModelForCausalLM

    configure_reference_math(torch)

    runtime = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    if not torch.cuda.is_available() or runtime.get_device_count() < 1:
        raise RuntimeError("需要 PyTorch 与 LLAISYS 都可访问的 NVIDIA GPU")
    runtime.set_device(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        tempfile.mkdtemp(prefix=f"{args.sample_id}_", dir=args.output_dir)
    ).resolve()
    hf_dir, native_dir = run_dir / "hf", run_dir / "native"
    hf_dir.mkdir()
    native_dir.mkdir()
    print("本次诊断目录：", run_dir, flush=True)

    captured = {}
    source_dtypes = {}

    def capture(name):
        def hook(_module, _inputs, output):
            if output.ndim != 3 or output.shape[0] != 1:
                raise ValueError(f"{name} 的 HF 输出不是 batch=1 的 [1,T,H]")
            source_dtypes[name] = str(output.dtype)
            if name.startswith("final."):
                output = output[:, -1:, :]
            # Only remove batch; all three checkpoints share [token, hidden].
            captured[name] = (
                output.detach().squeeze(0).float().cpu().contiguous().clone()
            )

        return hook

    print(f"加载 HF，采集 {len(checkpoints)} 个 prefill 检查点……", flush=True)
    hf_model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        )
        .to("cuda:0")
        .eval()
    )
    handles = [
        hf_model.model.embed_tokens.register_forward_hook(capture("embedding")),
        hf_model.model.layers[0].input_layernorm.register_forward_hook(
            capture("layer0.input_norm")
        ),
        hf_model.model.layers[0].register_forward_hook(capture("layer0.output")),
    ]
    from transformers.models.qwen2 import modeling_qwen2

    original_rope = modeling_qwen2.apply_rotary_pos_emb
    if args.full_layer:
        handles.append(hf_model.model.norm.register_forward_hook(capture("final.norm")))
        handles.append(hf_model.lm_head.register_forward_hook(capture("final.logits")))
        attn = hf_model.model.layers[0].self_attn
        mlp = hf_model.model.layers[0].mlp
        for name, module in (
            ("q_proj", attn.q_proj),
            ("k_proj", attn.k_proj),
            ("v_proj", attn.v_proj),
            ("gate", mlp.gate_proj),
            ("up", mlp.up_proj),
            ("down", mlp.down_proj),
        ):
            handles.append(module.register_forward_hook(capture("layer0." + name)))
        handles.append(
            hf_model.model.layers[0].post_attention_layernorm.register_forward_hook(
                capture("layer0.mlp_norm")
            )
        )

        def capture_input(name):
            def hook(module, inputs):
                capture(name)(module, (), inputs[0])

            return hook

        handles.append(
            attn.o_proj.register_forward_pre_hook(capture_input("layer0.attention"))
        )
        handles.append(
            hf_model.model.layers[0].post_attention_layernorm.register_forward_pre_hook(
                capture_input("layer0.post_attn")
            )
        )
        handles.append(
            mlp.down_proj.register_forward_pre_hook(capture_input("layer0.swiglu"))
        )

        def capture_rope(*positional, **kwargs):
            q, k = original_rope(*positional, **kwargs)
            if "layer0.q_rope" not in captured:
                for name, value in (("layer0.q_rope", q), ("layer0.k_rope", k)):
                    source_dtypes[name] = str(value.dtype)
                    captured[name] = (
                        value.detach()
                        .squeeze(0)
                        .permute(1, 0, 2)
                        .float()
                        .cpu()
                        .contiguous()
                    )
            return q, k

        modeling_qwen2.apply_rotary_pos_emb = capture_rope
    try:
        with torch.inference_mode():
            output = hf_model(
                input_ids=torch.tensor([ids], dtype=torch.long, device="cuda:0"),
                past_key_values=None,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            hf_token = int(output.logits[0, -1].argmax().item())
        for name in checkpoints:
            torch.save(captured[name], hf_dir / f"{name}.pt")
    finally:
        for handle in handles:
            handle.remove()
        modeling_qwen2.apply_rotary_pos_emb = original_rope
    if args.full_layer:
        del attn, mlp, module
    del output, hf_model, handles
    gc.collect()
    torch.cuda.empty_cache()

    print("加载 LLAISYS，开启本次 prefill 导出……", flush=True)
    model = llaisys.models.Qwen2(args.model, llaisys.DeviceType.NVIDIA)
    variable = "LLAISYS_QWEN2_DEBUG_DIR"
    previous = os.environ.get(variable)
    try:
        os.environ[variable] = str(native_dir)
        output_ids = model.generate(ids, max_new_tokens=1, top_k=1)
        runtime.device_synchronize()
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous
    if output_ids[: len(ids)] != ids or len(output_ids) != len(ids) + 1:
        raise RuntimeError("LLAISYS 单 token 生成结果格式不正确")

    results = {}
    lines = [
        "| checkpoint | shape | 相同元素比例 | 最大绝对误差 | 平均绝对误差 | RMSE | 非有限数 HF/native |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in checkpoints:
        native = read_native_tensor(native_dir, name)
        stats = tensor_errors(captured[name], native)
        results[name] = stats
        values = [
            "n/a" if stats[key] is None else f"{stats[key]:.8g}"
            for key in ("max_abs_error", "mean_abs_error", "rmse")
        ]
        lines.append(
            f"| {name} | {stats['shape']} | {stats['equal_fraction']:.2%} | "
            + " | ".join(values)
            + f" | {stats['hf_nonfinite']}/{stats['native_nonfinite']} |"
        )
    logit_analysis = None
    if args.full_layer:
        ref = captured["final.logits"].flatten()
        native = read_native_tensor(native_dir, "final.logits").flatten()

        def top(t):
            values, indices = t.topk(5)
            return {
                "ids": indices.tolist(),
                "logits": values.tolist(),
                "top1_top2_margin": float(values[0] - values[1]),
            }

        logit_analysis = {
            "hf": top(ref),
            "native": top(native),
            "cosine_similarity": float(
                torch.nn.functional.cosine_similarity(ref, native, dim=0)
            ),
        }
        print("最终 logits 的 top5 / margin / cosine：", logit_analysis, flush=True)
    model.close()
    summary = {
        "logit_analysis": logit_analysis,
        "hf_logits_to_keep": 1,
        "sample": sample,
        "model": str(args.model.resolve()),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "hf_attention": "eager",
        "hf_original_dtypes": source_dtypes,
        "tf32": False,
        "reduced_precision_gemm_reduction": False,
        "normalised_layout": "token,hidden; RoPE: token,head,dim",
        "comparison_dtype": "float32",
        "hf_first_token": hf_token,
        "native_first_token": output_ids[-1],
        "first_token_matches": hf_token == output_ids[-1],
        "checkpoints": results,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    table = "\n".join(lines) + "\n"
    (run_dir / "summary.md").write_text(table, encoding="utf-8")
    print(table, flush=True)
    print(f"首 token：HF={hf_token}, LLAISYS={output_ids[-1]}", flush=True)
    print(
        "首个非完全相同的检查点：",
        next((name for name in checkpoints if not results[name]["values_equal"]), "无"),
        flush=True,
    )
    print("这张表描述数值差异，不把首次舍入差异自动判定为 bug。", flush=True)
    print("完整张量、误差与首个差异坐标已保存：", run_dir, flush=True)


if __name__ == "__main__":
    main()
