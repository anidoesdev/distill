"""Generate a deployment decision report from quantization benchmark results.

Reads data/eval/quantization_benchmark.json and prints a structured
tradeoff analysis with a recommended deployment target.

Usage:
    python scripts/report_quantization.py
    python scripts/report_quantization.py --benchmark data/eval/quantization_benchmark.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCHMARK_PATH = Path("data/eval/quantization_benchmark.json")

# Minimum acceptable quality — anything below this fails the deployment bar
MIN_MACRO_F1 = 0.70
# Minimum throughput for production (tokens/sec at batch size 1)
MIN_TOKENS_PER_SEC = 30.0


def decide(results: dict[str, dict]) -> str:
    """Return the recommended deployment variant based on quality + throughput."""
    candidates = []
    for variant, r in results.items():
        f1 = r.get("macro_f1")
        tps = r.get("avg_tokens_per_sec", 0)
        if f1 is None or f1 < MIN_MACRO_F1:
            continue
        if tps < MIN_TOKENS_PER_SEC:
            continue
        candidates.append((variant, f1, tps))

    if not candidates:
        return "none — no variant meets minimum quality and throughput thresholds"

    # Prefer highest throughput among quality-passing variants
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][0]


def print_report(results: dict[str, dict]) -> None:
    print("\n" + "=" * 62)
    print("QUANTIZATION DEPLOYMENT REPORT")
    print("=" * 62)
    print(f"  Quality threshold:    macro F1 ≥ {MIN_MACRO_F1:.0%}")
    print(f"  Throughput threshold: ≥ {MIN_TOKENS_PER_SEC:.0f} tokens/sec")
    print()

    for variant, r in results.items():
        f1 = r.get("macro_f1")
        tps = r.get("avg_tokens_per_sec", 0)
        size = r.get("size_gb", 0)
        vram = r.get("vram_reserved_gb")
        latency = r.get("avg_latency_s", 0)

        f1_ok = f1 is not None and f1 >= MIN_MACRO_F1
        tps_ok = tps >= MIN_TOKENS_PER_SEC

        status = "PASS" if (f1_ok and tps_ok) else "FAIL"
        f1_str = f"{f1:.1%}" if f1 is not None else "n/a"
        vram_str = f"{vram:.1f} GB" if vram is not None else "n/a"

        print(f"  [{status}] {variant}")
        print(f"    macro F1:   {f1_str}  {'✓' if f1_ok else '✗'}")
        print(f"    tok/sec:    {tps:.1f}  {'✓' if tps_ok else '✗'}")
        print(f"    latency:    {latency:.2f}s  |  size: {size:.1f} GB  |  VRAM: {vram_str}")
        print()

    recommendation = decide(results)
    print(f"  Recommended deployment target: {recommendation}")
    print()

    if "bf16" in results and recommendation != "bf16":
        bf16_tps = results["bf16"].get("avg_tokens_per_sec", 1)
        rec_tps = results.get(recommendation, {}).get("avg_tokens_per_sec", 0)
        if rec_tps and bf16_tps:
            speedup = rec_tps / bf16_tps
            print(f"  Speedup over BF16: {speedup:.1f}×")

        bf16_f1 = results["bf16"].get("macro_f1")
        rec_f1 = results.get(recommendation, {}).get("macro_f1")
        if bf16_f1 and rec_f1:
            quality_gap = bf16_f1 - rec_f1
            print(f"  Quality cost:      {quality_gap:.1%} macro F1")

    print("=" * 62)
    print("\nNext: configure vLLM for the selected variant (session 22)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=str(BENCHMARK_PATH))
    args = parser.parse_args()

    path = Path(args.benchmark)
    if not path.exists():
        print(f"Benchmark results not found: {path}")
        print("Run: python scripts/benchmark_quantization.py")
        return

    results = json.loads(path.read_text())
    print_report(results)


if __name__ == "__main__":
    main()
