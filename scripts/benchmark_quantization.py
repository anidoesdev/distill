"""Benchmark all model variants: BF16, AWQ, GGUF.

Measures inference latency, throughput (tokens/sec), and VRAM usage for each
available model checkpoint. Loads existing eval F1 scores alongside performance
metrics to produce the full quality-vs-speed tradeoff table.

Writes results to data/eval/quantization_benchmark.json.

Usage:
    python scripts/benchmark_quantization.py
    python scripts/benchmark_quantization.py --variants bf16 awq
    python scripts/benchmark_quantization.py --n-prompts 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

RESULTS_DIR = Path("data/eval")

# Default paths — only variants with existing directories will be benchmarked
VARIANT_PATHS = {
    "bf16":   Path("checkpoints/dpo-merged"),
    "nf4":    Path("checkpoints/dpo"),        # original 4-bit NF4 from training
    "awq":    Path("checkpoints/awq"),
    "gguf":   Path("checkpoints/gguf"),
}

# Representative prompts for benchmarking — mix of short and long inputs
BENCH_PROMPTS = [
    "We trained a convolutional neural network on ImageNet using SGD with momentum 0.9. "
    "Top-1 accuracy reached 73.4%. Authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton.",

    "This study investigated the effect of learning rate schedules on transformer training. "
    "We compared cosine annealing, linear decay, and constant learning rates across three datasets: "
    "CIFAR-10, CIFAR-100, and TinyImageNet. Results indicate cosine annealing yields 2.3% higher "
    "accuracy on average. Statistical significance was confirmed via paired t-test (p < 0.05). "
    "Limitation: experiments were restricted to vision transformers only.",

    "Authors: Maria Chen, David Park. We propose a novel attention mechanism that reduces "
    "quadratic complexity to O(n log n). Our method, LinearAttention, was evaluated on the "
    "Long Range Arena benchmark. Key findings: 15% faster inference, 0.5% accuracy drop. "
    "Datasets: LRA-Text, LRA-Image, LRA-Pathfinder.",
]


def load_eval_f1(label: str) -> float | None:
    """Load macro F1 from a saved eval results file."""
    path = RESULTS_DIR / f"{label}_eval_results.json"
    if not path.exists():
        # Try common naming conventions
        for candidate in ["dpo_eval_results.json", "sft_eval_results.json"]:
            p = RESULTS_DIR / candidate
            if p.exists() and label in candidate:
                path = p
                break
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("metrics", {}).get("macro_f1")


def benchmark_hf(model_dir: Path, load_in_4bit: bool, prompts: list[str]) -> dict:
    """Benchmark a HuggingFace model (BF16 or NF4)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from extractor.prompt import build_messages

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), quantization_config=bnb, device_map="auto", trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    model.eval()
    device = next(model.parameters()).device

    vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    latencies, token_counts = [], []
    for prompt_text in prompts:
        messages = build_messages(prompt_text)
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - t0

        n_new = output_ids.shape[1] - input_ids.shape[1]
        latencies.append(elapsed)
        token_counts.append(n_new)

    avg_tokens = sum(token_counts) / len(token_counts)
    avg_latency = sum(latencies) / len(latencies)
    tokens_per_sec = avg_tokens / avg_latency if avg_latency > 0 else 0.0

    # Report reserved memory (more stable than allocated)
    vram_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "avg_latency_s": round(avg_latency, 3),
        "avg_tokens_per_sec": round(tokens_per_sec, 1),
        "avg_output_tokens": round(avg_tokens, 1),
        "vram_allocated_gb": round(vram_gb, 2),
        "vram_reserved_gb": round(vram_reserved, 2),
    }


def benchmark_awq(model_dir: Path, prompts: list[str]) -> dict:
    """Benchmark an AWQ-quantized model."""
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise ImportError("autoawq not installed. Run: pip install autoawq")

    import torch
    from transformers import AutoTokenizer

    from extractor.prompt import build_messages

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoAWQForCausalLM.from_quantized(str(model_dir), fuse_layers=True)
    model.eval()

    import torch
    vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    latencies, token_counts = [], []
    for prompt_text in prompts:
        messages = build_messages(prompt_text)
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - t0
        n_new = output_ids.shape[1] - inputs.shape[1]
        latencies.append(elapsed)
        token_counts.append(n_new)

    avg_tokens = sum(token_counts) / len(token_counts)
    avg_latency = sum(latencies) / len(latencies)
    vram_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "avg_latency_s": round(avg_latency, 3),
        "avg_tokens_per_sec": round(avg_tokens / avg_latency, 1) if avg_latency > 0 else 0.0,
        "avg_output_tokens": round(avg_tokens, 1),
        "vram_allocated_gb": round(vram_gb, 2),
        "vram_reserved_gb": round(vram_reserved, 2),
    }


def benchmark_gguf(gguf_path: Path, prompts: list[str]) -> dict:
    """Benchmark a GGUF model using llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python not installed.\n"
            "  pip install llama-cpp-python  # CPU-only\n"
            "  CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python  # CUDA"
        )

    from extractor.prompt import EXTRACTION_SYSTEM_PROMPT

    # n_gpu_layers=-1 offloads all layers to GPU if CUDA build
    llm = Llama(model_path=str(gguf_path), n_gpu_layers=-1, n_ctx=1024, verbose=False)

    latencies, token_counts = [], []
    for prompt_text in prompts:
        prompt = f"<|im_start|>system\n{EXTRACTION_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

        t0 = time.perf_counter()
        output = llm(prompt, max_tokens=256, temperature=0.0, echo=False)
        elapsed = time.perf_counter() - t0

        n_tokens = output["usage"]["completion_tokens"]
        latencies.append(elapsed)
        token_counts.append(n_tokens)

    avg_tokens = sum(token_counts) / len(token_counts)
    avg_latency = sum(latencies) / len(latencies)

    del llm

    return {
        "avg_latency_s": round(avg_latency, 3),
        "avg_tokens_per_sec": round(avg_tokens / avg_latency, 1) if avg_latency > 0 else 0.0,
        "avg_output_tokens": round(avg_tokens, 1),
        "vram_allocated_gb": None,   # llama-cpp-python doesn't expose CUDA memory easily
        "vram_reserved_gb": None,
    }


def print_table(results: dict[str, dict]) -> None:
    if not results:
        print("No results to display.")
        return

    labels = list(results.keys())
    w = 14

    print("\n" + "=" * (24 + w * len(labels) + 2 * len(labels)))
    print("QUANTIZATION BENCHMARK")
    print("=" * (24 + w * len(labels) + 2 * len(labels)))

    header = f"  {'':22}"
    for label in labels:
        header += f"  {label:>{w}}"
    print(header)
    print()

    def row(name: str, key: str, fmt: str = ".1f", suffix: str = "") -> None:
        line = f"  {name:<22}"
        for label in labels:
            val = results[label].get(key)
            if val is None:
                line += f"  {'n/a':>{w}}"
            elif fmt == "%":
                line += f"  {val:>{w}.1%}"
            else:
                line += f"  {(str(round(val, 2)) + suffix):>{w}}"
        print(line)

    row("macro F1",          "macro_f1",           fmt="%")
    print()
    row("tokens/sec",        "avg_tokens_per_sec", suffix=" tok/s")
    row("avg latency (s)",   "avg_latency_s",      suffix="s")
    row("VRAM reserved (GB)", "vram_reserved_gb",  suffix=" GB")
    row("size on disk (GB)", "size_gb",            suffix=" GB")

    print("=" * (24 + w * len(labels) + 2 * len(labels)))

    # Speedup row vs BF16
    if "bf16" in results and results["bf16"].get("avg_tokens_per_sec"):
        bf16_tps = results["bf16"]["avg_tokens_per_sec"]
        print("\n  Speedup vs BF16:")
        for label in labels:
            if label == "bf16":
                continue
            tps = results[label].get("avg_tokens_per_sec")
            if tps:
                speedup = tps / bf16_tps
                print(f"    {label:<10} {speedup:.1f}×")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants", nargs="+",
        default=None,
        choices=["bf16", "nf4", "awq", "gguf"],
        help="Which variants to benchmark. Default: all that exist.",
    )
    parser.add_argument("--n-prompts", type=int, default=len(BENCH_PROMPTS))
    parser.add_argument("--output", default=str(RESULTS_DIR / "quantization_benchmark.json"))
    args = parser.parse_args()

    prompts = BENCH_PROMPTS[:args.n_prompts]
    results: dict[str, dict] = {}

    variants_to_run = args.variants or list(VARIANT_PATHS.keys())

    for variant in variants_to_run:
        path = VARIANT_PATHS[variant]

        if variant == "gguf":
            # Find the .gguf file in the gguf directory
            from training.config import QuantizationConfig
            gguf_type = QuantizationConfig().gguf_type
            gguf_file = path / f"model-{gguf_type}.gguf"
            if not gguf_file.exists():
                print(f"  [{variant}] not found: {gguf_file} — skipping")
                continue
            actual_path = gguf_file
        else:
            actual_path = path
            if not actual_path.exists():
                print(f"  [{variant}] not found: {actual_path} — skipping")
                continue

        print(f"  [{variant}] benchmarking {actual_path}...")
        try:
            if variant == "awq":
                perf = benchmark_awq(actual_path, prompts)
            elif variant == "gguf":
                perf = benchmark_gguf(actual_path, prompts)
            else:
                load_in_4bit = (variant == "nf4")
                perf = benchmark_hf(actual_path, load_in_4bit, prompts)

            # Size on disk
            size_gb = sum(f.stat().st_size for f in actual_path.parent.rglob("*") if f.is_file()) / 1e9
            if variant == "gguf":
                size_gb = actual_path.stat().st_size / 1e9

            # Load eval F1 if available
            macro_f1 = load_eval_f1(variant)

            results[variant] = {
                "model_path": str(actual_path),
                "size_gb": round(size_gb, 2),
                "macro_f1": macro_f1,
                **perf,
            }
            tps = perf.get("avg_tokens_per_sec", 0)
            f1_str = f"  macro_f1={macro_f1:.3f}" if macro_f1 else ""
            print(f"    {tps:.1f} tok/s{f1_str}")

        except Exception as exc:
            print(f"    FAILED: {exc}")

    print_table(results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nBenchmark results → {out}")
    print("Next: python scripts/report_quantization.py")


if __name__ == "__main__":
    main()
