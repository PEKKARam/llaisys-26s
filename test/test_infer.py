import gc
from test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import time
import llaisys
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def load_hf_model(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path


def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs[0].tolist(), result, inputs.shape[-1]


def load_llaisys_model(model_path, device_name):
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    return model


def llaisys_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    return outputs, tokenizer.decode(outputs, skip_special_tokens=True), len(inputs)


def synchronize(device_name):
    if device_name == "nvidia":
        torch.cuda.synchronize()


def print_timing(name, load_seconds, infer_seconds, prompt_tokens, total_tokens):
    output_tokens = max(0, total_tokens - prompt_tokens)
    tokens_per_second = output_tokens / infer_seconds if infer_seconds > 0 else 0.0
    print(f"{name} load: {load_seconds * 1000:.2f} ms")
    print(f"{name} end-to-end: {infer_seconds * 1000:.2f} ms")
    print(
        f"{name} tokens: prompt={prompt_tokens}, output={output_tokens}, "
        f"throughput={tokens_per_second:.3f} output tok/s\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    llaisys_runtime = llaisys.RuntimeAPI(llaisys_device(args.device))
    if llaisys_runtime.get_device_count() < 1:
        raise SystemExit(
            f"No LLAISYS {args.device} device is available; build and install the "
            "requested backend before running inference tests"
        )
    llaisys_runtime.set_device(0)

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    load_start = time.perf_counter()
    tokenizer, model, model_path = load_hf_model(args.model, args.device)
    synchronize(args.device)
    hf_load_seconds = time.perf_counter() - load_start

    # Example prompt
    synchronize(args.device)
    start_time = time.perf_counter()
    tokens, output, hf_prompt_tokens = hf_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    synchronize(args.device)
    hf_infer_seconds = time.perf_counter() - start_time

    del model
    gc.collect()

    print("\n=== Answer ===\n")
    print("Tokens:")
    print(tokens)
    print("\nContents:")
    print(output)
    print("\n")
    print_timing(
        "Hugging Face",
        hf_load_seconds,
        hf_infer_seconds,
        hf_prompt_tokens,
        len(tokens),
    )

    load_start = time.perf_counter()
    model = load_llaisys_model(model_path, args.device)
    llaisys_runtime.device_synchronize()
    llaisys_load_seconds = time.perf_counter() - load_start
    start_time = time.perf_counter()
    llaisys_tokens, llaisys_output, llaisys_prompt_tokens = llaisys_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )

    llaisys_runtime.device_synchronize()
    llaisys_infer_seconds = time.perf_counter() - start_time

    print("\n=== Your Result ===\n")
    print("Tokens:")
    print(llaisys_tokens)
    print("\nContents:")
    print(llaisys_output)
    print("\n")
    print_timing(
        "LLAISYS",
        llaisys_load_seconds,
        llaisys_infer_seconds,
        llaisys_prompt_tokens,
        len(llaisys_tokens),
    )

    if args.test:
        assert llaisys_tokens == tokens
        print("\033[92mTest passed!\033[0m\n")
