"""Read one saved prompt and run a short LLAISYS generation."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))


def load_sample(path, sample_id):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                sample = json.loads(line)
                if sample["id"] == sample_id:
                    return sample
    raise ValueError(f"找不到样本 {sample_id}，请检查 {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default="row_3_len_16")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "nvidia"), default="nvidia")
    parser.add_argument(
        "--model", type=Path, default=REPO_ROOT / "DeepSeek-R1-Distill-Qwen-1.5B"
    )
    parser.add_argument(
        "--prompts", type=Path,
        default=REPO_ROOT / "cache/prompt_sets/wikitext2_smoke.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "benchmark/results/generation_smoke.jsonl",
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens 必须大于 0")

    sample = load_sample(args.prompts, args.sample_id)
    if sample["input_mode"] != "raw_completion":
        raise ValueError("此脚本只运行 raw_completion 样本")
    input_ids = sample["input_ids"]
    if not input_ids or len(input_ids) != sample["prompt_tokens"]:
        raise ValueError("样本的 input_ids 长度与 prompt_tokens 不一致或为空")
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    if any(type(token) is not int or not 0 <= token < config["vocab_size"]
           for token in input_ids):
        raise ValueError("输入中存在非法 token ID")
    if len(input_ids) + args.max_new_tokens > config["max_position_embeddings"]:
        raise ValueError("请求长度超过模型的最大序列长度")

    # 前面只读取文件；到这里才加载后端和 tokenizer。
    import llaisys
    from transformers import AutoTokenizer

    device = (llaisys.DeviceType.NVIDIA if args.device == "nvidia"
              else llaisys.DeviceType.CPU)
    runtime = llaisys.RuntimeAPI(device)
    if runtime.get_device_count() < 1:
        raise RuntimeError(f"没有可用的 {args.device} 后端或设备")
    runtime.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    print(f"样本：{sample['id']}，输入长度：{len(input_ids)}", flush=True)
    print("输入文本：", tokenizer.decode(input_ids), flush=True)
    print("正在加载模型……", flush=True)
    model = llaisys.models.Qwen2(args.model, device)

    # 直接使用保存的整数列表，不把文本重新分词，也不添加聊天模板。
    output_ids = model.generate(
        input_ids, max_new_tokens=args.max_new_tokens, top_k=1
    )
    runtime.device_synchronize()

    # generate 返回“原输入 + 新 token”，切掉前面这段才能得到新增部分。
    if output_ids[:len(input_ids)] != input_ids:
        raise RuntimeError("模型返回结果没有保留原输入前缀")
    generated_ids = output_ids[len(input_ids):]
    if not 1 <= len(generated_ids) <= args.max_new_tokens:
        raise RuntimeError("模型返回的新增 token 数量不符合请求")
    if any(not 0 <= token < config["vocab_size"] for token in generated_ids):
        raise RuntimeError("模型返回了非法 token ID")
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    result = {
        **sample,
        "run_type": "generation_smoke",
        "model": str(args.model.resolve()),
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": len(generated_ids),
        "generated_ids": generated_ids,
        "output_ids": output_ids,
        "generated_text": generated_text,
        "reference_check": "not_run",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("输入 token IDs：", input_ids)
    print("新增 token IDs：", generated_ids)
    print("新增文本：", repr(generated_text))
    print(f"实际生成 {len(generated_ids)} 个 token（遇到 EOS 可以提前结束）")
    print(f"结果已追加到：{args.output}")
    print("本次是链路检查；尚未进行 HF 对照或正式性能测量。")


if __name__ == "__main__":
    main()
