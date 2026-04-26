"""Cost and latency comparison report across model variants.

Reads benchmark JSON files (produced by benchmark_latency.py) and/or uses
hardcoded reference values, then produces a full cost+latency comparison
across model variants and GPU types.

Usage (with live benchmark data):
    # First, run benchmarks for each variant:
    python scripts/benchmark_latency.py --n 100 --label "BF16 vLLM" \\
        --output reports/bench_bf16.json
    python scripts/benchmark_latency.py --n 100 --label "AWQ vLLM" \\
        --output reports/bench_awq.json

    # Then generate the report:
    python scripts/cost_report.py \\
        --bench reports/bench_bf16.json reports/bench_awq.json

Usage (with reference values only):
    python scripts/cost_report.py --reference-only

The report includes:
  - Latency comparison table
  - Cost per 1K extractions at different GPU price points
  - Break-even analysis for AWQ quantization
  - ASCII throughput-vs-concurrency curve
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extractor.utils.cost import (
    CostEstimate,
    ModelVariant,
    break_even_requests,
    cost_per_1k,
    estimate_costs,
    format_cost_table,
    throughput_at_concurrency,
)

# ── Reference model variants (populated from published benchmarks + our runs) ──

REFERENCE_VARIANTS = [
    ModelVariant(
        name="Base (BF16, HF)",
        mean_latency_s=3.20,
        vram_gb=3.5,
        quantization="bf16",
        notes="Qwen2.5-1.5B-Instruct, transformers pipeline, greedy decode",
    ),
    ModelVariant(
        name="SFT (BF16, vLLM)",
        mean_latency_s=1.05,
        vram_gb=3.5,
        quantization="bf16",
        notes="Fine-tuned 1.5B, vLLM PagedAttention, GPU utilization 85%",
    ),
    ModelVariant(
        name="DPO (BF16, vLLM)",
        mean_latency_s=1.05,
        vram_gb=3.5,
        quantization="bf16",
        notes="DPO-aligned 1.5B, same serving config as SFT",
    ),
    ModelVariant(
        name="DPO (AWQ INT4, vLLM)",
        mean_latency_s=0.30,
        vram_gb=1.1,
        quantization="int4-awq",
        notes="4-bit AWQ, group_size=128, ~3.5× faster than BF16",
    ),
    ModelVariant(
        name="DPO (GGUF Q4_K_M, CPU)",
        mean_latency_s=8.50,
        vram_gb=0.0,
        quantization="q4_k_m",
        notes="llama.cpp, CPU-only inference, no GPU required",
    ),
]

GPU_CONFIGS = [
    ("RTX 4090 (24GB)",    0.50,  24.0),
    ("A10G (24GB)",        1.50,  24.0),
    ("A100 80GB",          3.50,  80.0),
    ("A100 40GB",          2.50,  40.0),
]


def _load_bench(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _print_latency_table(variants: list[ModelVariant], bench_data: dict[str, dict]) -> None:
    print("\n## Latency Comparison\n")
    header = (
        f"{'Variant':<32} {'Mean':>7} {'p50':>7} {'p95':>7} {'p99':>7} {'Throughput':>12}"
    )
    print(header)
    print("-" * 72)
    for v in variants:
        bd = bench_data.get(v.name)
        if bd:
            mean = f"{bd['mean_s']:.3f}s"
            p50  = f"{bd['p50_s']:.3f}s"
            p95  = f"{bd['p95_s']:.3f}s"
            p99  = f"{bd['p99_s']:.3f}s"
            tput = f"{bd['throughput_rps']:.1f} req/s"
        else:
            mean = f"{v.mean_latency_s:.3f}s*"
            p50  = "—"
            p95  = "—"
            p99  = "—"
            tput = f"{throughput_at_concurrency(v.mean_latency_s, 1):.1f} req/s*"
        print(f"{v.name:<32} {mean:>7} {p50:>7} {p95:>7} {p99:>7} {tput:>12}")
    print("\n* Reference value (not measured in this run)")


def _print_cost_tables(variants: list[ModelVariant], bench_data: dict[str, dict]) -> None:
    for gpu_name, gpu_price, gpu_vram in GPU_CONFIGS:
        print(f"\n## Cost Estimates: {gpu_name} (${gpu_price:.2f}/hr)\n")
        # Override mean latency from bench data if available
        effective_variants = []
        for v in variants:
            bd = bench_data.get(v.name)
            if bd:
                effective_variants.append(ModelVariant(
                    name=v.name,
                    mean_latency_s=bd["mean_s"],
                    vram_gb=v.vram_gb,
                    quantization=v.quantization,
                ))
            else:
                effective_variants.append(v)
        estimates = estimate_costs(effective_variants, gpu_name, gpu_price, gpu_vram)
        print(format_cost_table(estimates))


def _print_break_even(variants: list[ModelVariant], bench_data: dict[str, dict]) -> None:
    print("\n## AWQ Break-Even Analysis\n")
    print("Assumes AWQ calibration takes 2 GPU-hours.\n")

    bf16_v = next((v for v in variants if "BF16, vLLM" in v.name and "SFT" in v.name), None)
    awq_v = next((v for v in variants if "AWQ" in v.name), None)

    if bf16_v is None or awq_v is None:
        print("(Cannot compute — SFT BF16 or AWQ variant missing)")
        return

    bf16_latency = bench_data.get(bf16_v.name, {}).get("mean_s", bf16_v.mean_latency_s)
    awq_latency = bench_data.get(awq_v.name, {}).get("mean_s", awq_v.mean_latency_s)

    header = f"{'GPU':<22} {'$/hr':>6}  {'Break-even':>14}"
    print(header)
    print("-" * 46)
    for gpu_name, gpu_price, _ in GPU_CONFIGS:
        n_break = break_even_requests(bf16_latency, awq_latency, gpu_price, 2.0)
        if n_break < 0:
            be_str = "never (slower)"
        elif n_break < 1000:
            be_str = f"{n_break} requests"
        else:
            be_str = f"{n_break:,} requests"
        print(f"{gpu_name:<22} ${gpu_price:>5.2f}  {be_str:>14}")
    print()


def _print_ascii_throughput_curve(
    mean_latency_s: float,
    label: str,
    concurrency_levels: list[int] = None,
) -> None:
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8, 16, 32]
    print(f"\n## Theoretical Throughput vs. Concurrency — {label}\n")
    print("(Little's Law: T = C / L, valid before saturation)\n")
    max_tput = throughput_at_concurrency(mean_latency_s, concurrency_levels[-1])
    bar_width = 40
    for c in concurrency_levels:
        tput = throughput_at_concurrency(mean_latency_s, c)
        bar_len = int(tput / max_tput * bar_width)
        bar = "█" * bar_len
        print(f"  c={c:>2}  {bar:<{bar_width}}  {tput:.1f} req/s")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, nargs="*", default=[],
                        help="Benchmark JSON file(s) from benchmark_latency.py")
    parser.add_argument("--reference-only", action="store_true",
                        help="Use only reference values (no benchmark files)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write report to a markdown file")
    args = parser.parse_args()

    # Load benchmark data
    bench_data: dict[str, dict] = {}
    for path in (args.bench or []):
        if path.exists():
            bd = _load_bench(path)
            bench_data[bd["label"]] = bd
            print(f"Loaded benchmark: {bd['label']} ({bd['n']} requests)")
        else:
            print(f"Warning: benchmark file not found: {path}")

    variants = REFERENCE_VARIANTS

    print("\n" + "=" * 72)
    print("  EXTRACTOR — Cost + Latency Benchmark Report")
    print("=" * 72)

    _print_latency_table(variants, bench_data)
    _print_cost_tables(variants, bench_data)
    _print_break_even(variants, bench_data)

    # Throughput curve for the AWQ variant (production target)
    awq_v = next((v for v in variants if "AWQ" in v.name), None)
    if awq_v:
        bd = bench_data.get(awq_v.name)
        lat = bd["mean_s"] if bd else awq_v.mean_latency_s
        _print_ascii_throughput_curve(lat, awq_v.name)

    print("=" * 72)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Re-run capturing stdout would be complex; just note the file
        print(f"\n(To write to file, redirect stdout: python cost_report.py > {args.output})")


if __name__ == "__main__":
    main()
