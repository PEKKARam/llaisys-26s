"""Compare native Qwen2 and Hugging Face greedy outputs on saved token IDs."""

import argparse
import gc
import hashlib
import json
import os
import sys
import sysconfig
from itertools import zip_longest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))


def configure_reference_math(torch):
    """Use FP32 GEMM accumulation rather than hardware-specific reduced partial sums."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def configure_local_python_headers():
    """Allow Triton's C compiler to use locally extracted Python headers."""
    system_header = Path(sysconfig.get_paths()["include"]) / "Python.h"
    local_include = REPO_ROOT / "cache/python-dev/usr/include"
    python_include = (
        local_include / f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    if not system_header.exists() and (python_include / "Python.h").exists():
        paths = [str(python_include), str(local_include)]
        if os.environ.get("C_INCLUDE_PATH"):
            paths.append(os.environ["C_INCLUDE_PATH"])
        os.environ["C_INCLUDE_PATH"] = os.pathsep.join(paths)
        print(f"使用本地 Python 开发头文件：{python_include}", flush=True)


def hf_greedy(model, input_ids, max_new_tokens, eos_token_id, torch, logits_to_keep=1):
    """One prefill followed by cached, single-token decode; no sampling."""
    current = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    past = None
    generated = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output = model(
                input_ids=current,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                logits_to_keep=logits_to_keep,
            )
            next_token = int(output.logits[0, -1].argmax().item())
            generated.append(next_token)
            if next_token == eos_token_id:
                break
            past = output.past_key_values
            current = torch.tensor(
                [[next_token]], dtype=torch.long, device=model.device
            )
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-id", default="row_3_len_16")
    selection.add_argument(
        "--all", action="store_true", help="Compare all saved samples"
    )
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--hf-logits-to-keep", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--model", type=Path, default=REPO_ROOT / "DeepSeek-R1-Distill-Qwen-1.5B"
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=REPO_ROOT / "cache/prompt_sets/wikitext2_smoke.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmark/results/generation_compare.jsonl",
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens 必须大于 0")

    with args.prompts.open(encoding="utf-8") as file:
        samples = [json.loads(line) for line in file if line.strip()]
    if not args.all:
        samples = [sample for sample in samples if sample["id"] == args.sample_id]
    if not samples:
        parser.error("没有找到待比较的样本")
    if len({sample["id"] for sample in samples}) != len(samples):
        parser.error("样本 ID 重复")
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    for sample in samples:
        ids = sample["input_ids"]
        if (
            sample["input_mode"] != "raw_completion"
            or not ids
            or len(ids) != sample["prompt_tokens"]
        ):
            parser.error(f"样本格式或输入长度不正确：{sample['id']}")
        if any(
            type(token) is not int or not 0 <= token < config["vocab_size"]
            for token in ids
        ):
            parser.error(f"样本含非法 token：{sample['id']}")
        if len(ids) + args.max_new_tokens - 1 > config["max_position_embeddings"]:
            parser.error(f"请求超过模型序列上限：{sample['id']}")

    configure_local_python_headers()
    import torch
    import transformers
    import llaisys
    from transformers import AutoModelForCausalLM, AutoTokenizer

    configure_reference_math(torch)

    if not torch.cuda.is_available():
        raise RuntimeError("本对照脚本需要可用的 NVIDIA GPU")
    device = llaisys.DeviceType.NVIDIA
    runtime = llaisys.RuntimeAPI(device)
    if runtime.get_device_count() < 1:
        raise RuntimeError("LLAISYS NVIDIA 后端不可用，请先构建安装")
    runtime.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    eos_token_id = int(config["eos_token_id"])

    print("加载 HF：BF16、eager attention、greedy、KV Cache 开启", flush=True)
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
    hf_results = {}
    for sample in samples:
        hf_results[sample["id"]] = hf_greedy(
            hf_model,
            sample["input_ids"],
            args.max_new_tokens,
            eos_token_id,
            torch,
            args.hf_logits_to_keep,
        )
        print(f"HF 已完成：{sample['id']}", flush=True)
    # HF 与 LLAISYS 分别运行，避免两份权重和缓存同时占用显存。
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    print("加载 LLAISYS：BF16、argmax、每个请求重置 KV Cache", flush=True)
    native_model = llaisys.models.Qwen2(args.model, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    with args.output.open("a", encoding="utf-8") as file:
        for sample in samples:
            ids = sample["input_ids"]
            output = native_model.generate(
                ids, max_new_tokens=args.max_new_tokens, top_k=1
            )
            runtime.device_synchronize()
            if output[: len(ids)] != ids:
                raise RuntimeError("LLAISYS 返回结果没有保留输入前缀")
            native_ids = output[len(ids) :]
            hf_ids = hf_results[sample["id"]]
            mismatch = next(
                (
                    {
                        "new_token_index": i,
                        "hf_token": hf_token,
                        "llaisys_token": native_token,
                    }
                    for i, (hf_token, native_token) in enumerate(
                        zip_longest(hf_ids, native_ids)
                    )
                    if hf_token != native_token
                ),
                None,
            )
            matched = mismatch is None
            passed += int(matched)
            record = {
                **sample,
                "run_type": "greedy_token_comparison",
                "model": str(args.model.resolve()),
                "device": "nvidia",
                "dtype": "bf16",
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "hf_attention": "eager",
                "sampling": "greedy",
                "use_cache": True,
                "tf32": False,
                "reduced_precision_gemm_reduction": False,
                "hf_logits_to_keep": args.hf_logits_to_keep,
                "hf_disable_addmm_cuda_lt": os.environ.get(
                    "DISABLE_ADDMM_CUDA_LT", "0"
                ),
                "cuda_version": os.environ.get("LLAISYS_CUDA_VERSION", "2"),
                "engine_version": os.environ.get("LLAISYS_ENGINE_VERSION", "2"),
                "library_sha256": hashlib.sha256(
                    Path(llaisys.libllaisys.LIB_LLAISYS._name).read_bytes()
                ).hexdigest(),
                "eos_token_id": eos_token_id,
                "max_new_tokens": args.max_new_tokens,
                "hf_generated_ids": hf_ids,
                "llaisys_generated_ids": native_ids,
                "reference_check": "pass" if matched else "fail",
                "first_mismatch": mismatch,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            print(f"{'PASS' if matched else 'FAIL'} {sample['id']}", flush=True)
            print("  HF      ：", hf_ids, repr(tokenizer.decode(hf_ids)), flush=True)
            print(
                "  LLAISYS ：",
                native_ids,
                repr(tokenizer.decode(native_ids)),
                flush=True,
            )
            if mismatch:
                print("  首个差异（新增 token 从 0 编号）：", mismatch, flush=True)

    native_model.close()
    print(f"结果：{passed}/{len(samples)} 通过；记录已追加到 {args.output}", flush=True)
    print("这里只比较生成 token，不测性能，也不证明所有 logits 数值一致。", flush=True)
    if passed != len(samples):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
