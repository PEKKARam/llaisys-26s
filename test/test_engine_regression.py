"""Small, deterministic end-to-end/cache regression; no downloaded model needed."""

import argparse
import ctypes
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmark"))
from compare_prompt import configure_local_python_headers, configure_reference_math

configure_local_python_headers()
import torch
import llaisys
from transformers import Qwen2Config, Qwen2ForCausalLM
from llaisys.libllaisys import LIB_LLAISYS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "nvidia"), default="nvidia")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    configure_reference_math(torch)
    torch.manual_seed(20261102)
    device = (
        llaisys.DeviceType.NVIDIA if args.device == "nvidia" else llaisys.DeviceType.CPU
    )
    torch_device = "cuda" if args.device == "nvidia" else "cpu"
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        eos_token_id=63,
        tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config._attn_implementation = "eager"
    hf = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    tests = []

    def check(name, condition):
        assert condition, name
        tests.append(name)
        print("PASS", name, flush=True)

    def raises(name, fn, exception):
        try:
            fn()
        except exception:
            check(name, True)
        else:
            raise AssertionError(name + " did not fail")

    with tempfile.TemporaryDirectory(prefix="llaisys-tiny-") as directory:
        hf.save_pretrained(directory)
        hf = hf.to(torch_device)
        with llaisys.models.Qwen2(directory, device, cache_capacity=16) as native:
            ids = [1, 7, 4, 19, 23, 6, 11, 9, 8, 14, 3, 12, 2, 20, 21, 22]
            initial = native.state
            check(
                "cache footprint formula",
                initial["cache_bytes"] == 2 * 2 * 16 * 2 * 8 * 2,
            )
            check("initial cached length", initial["cached_len"] == 0)
            raises("decode before prefill", lambda: native.decode(1), RuntimeError)
            for length in (1, 2, 5, 11):
                with torch.inference_mode():
                    expected = int(
                        hf(
                            input_ids=torch.tensor([ids[:length]], device=torch_device),
                            use_cache=False,
                        )
                        .logits[0, -1]
                        .argmax()
                    )
                check(
                    f"HF tiny prefill length {length}",
                    native.prefill(ids[:length]) == expected,
                )
                check(f"cached length {length}", native.state["cached_len"] == length)
            native.prefill(ids[:3])
            incremental = native.infer(ids[3:7])
            check(
                "chunked prefill equals full prefill",
                incremental == native.prefill(ids[:7]),
            )
            native.prefill(ids[:4])
            for length in range(5, 10):
                incremental = native.decode(ids[length - 1])
                check(
                    f"cached decode equals full prefix {length}",
                    incremental == native.prefill(ids[:length]),
                )
            # Re-establish one cache then run sequential decode without resetting.
            native.prefill(ids[:4])
            for length in range(5, 10):
                result = native.decode(ids[length - 1])
                with torch.inference_mode():
                    expected = int(
                        hf(
                            input_ids=torch.tensor([ids[:length]], device=torch_device),
                            use_cache=False,
                        )
                        .logits[0, -1]
                        .argmax()
                    )
                check(f"HF tiny cached decode {length}", result == expected)
            native.prefill(ids)
            state = native.state
            check("capacity exactly full", state["cached_len"] == 16)
            raises("capacity overflow", lambda: native.decode(1), RuntimeError)
            check("overflow keeps length", native.state["cached_len"] == 16)
            raises("invalid vocabulary ID", lambda: native.prefill([64]), ValueError)
            check("invalid prefill keeps old cache", native.state["cached_len"] == 16)
            raises("empty infer", lambda: native.infer([]), ValueError)
            # Deliberately bypass Python validation: C API must return errors, not abort.
            invalid = (ctypes.c_int64 * 1)(-1)
            native.reset()
            check(
                "C API invalid input returns -1",
                LIB_LLAISYS.llaisysQwen2ModelInfer(native._model, invalid, 1) == -1,
            )
            check("C API exposes error text", bool(LIB_LLAISYS.llaisysQwen2LastError()))
            native.prefill(ids[:8])
            before = native.state
            for _ in range(3):
                native.prefill(ids[:8])
                native.decode(1)
            after = native.state
            check(
                "workspace allocations stable after warmup",
                after["workspace_allocations"] == before["workspace_allocations"],
            )
            native.reset()
            check(
                "reset retains cache capacity",
                native.state["cached_len"] == 0
                and native.state["cache_bytes"] == state["cache_bytes"],
            )
            output = native.generate(ids[:3], max_new_tokens=4)
            check(
                "generate output and cache lengths",
                len(output) == 7 and native.state["cached_len"] == 6,
            )
            check(
                "zero output request",
                native.generate(ids[:3], max_new_tokens=0) == ids[:3],
            )
        native.close()
        raises("use after close", lambda: native.prefill([1]), RuntimeError)
        # Tied embedding loading exercises the alternate weight path.
        hf = hf.cpu()
        hf.config.tie_word_embeddings = True
        hf.tie_weights()
        hf.save_pretrained(directory)
        with llaisys.models.Qwen2(directory, device, cache_capacity=8) as tied:
            check("tied embedding loads and infers", 0 <= tied.prefill([1, 2, 3]) < 64)
    report = {
        "device": args.device,
        "passed": len(tests),
        "tests": tests,
        "seed": 20261102,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(tests)} regression checks passed")


if __name__ == "__main__":
    main()
