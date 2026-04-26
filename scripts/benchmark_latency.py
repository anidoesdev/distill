"""Latency and throughput benchmark for the EXTRACTOR API.

Sends N requests at a given concurrency level and reports:
  p50, p90, p95, p99, max latency
  throughput (req/s)
  error rate

Usage:
    # Single-threaded baseline
    python scripts/benchmark_latency.py --n 100 --concurrency 1

    # Find saturation point
    python scripts/benchmark_latency.py --n 200 --concurrency 1
    python scripts/benchmark_latency.py --n 200 --concurrency 4
    python scripts/benchmark_latency.py --n 200 --concurrency 8
    python scripts/benchmark_latency.py --n 200 --concurrency 16

    # Save results for cost_report.py
    python scripts/benchmark_latency.py --n 100 --concurrency 1 \\
        --output reports/latency_c1.json --label "vLLM AWQ c=1"

Prerequisites:
    docker compose up   # vLLM + extractor-api must be running
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

from extractor.utils.cost import LatencyStats

SAMPLE_TEXTS = [
    (
        "Authors: Chen L, Zhang W, Patel R. We fine-tuned GPT-2 on arXiv abstracts "
        "using AdamW (lr=5e-5, batch=32, epochs=3). Evaluated on ROUGE-L and BERTScore. "
        "Dataset: arXiv (1.5M abstracts, 2010-2023). "
        "Key finding: ROUGE-L improved from 0.21 to 0.38 vs. zero-shot baseline. "
        "Limitation: Only evaluated on physics and CS domains. "
        "Statistical test: paired t-test (p < 0.001)."
    ),
    (
        "We present a meta-analysis of 47 studies on transformer attention mechanisms. "
        "Statistical significance assessed via Bonferroni correction (α=0.05/47). "
        "No single author attribution — consortium work. "
        "Datasets: ACL Anthology, PapersWithCode. "
        "Findings: Sparse attention achieves 94% of dense attention F1 at 3× speed. "
        "Limitations: High heterogeneity across studies (I²=0.71)."
    ),
    (
        "Authors: Müller A, Santos B, Kim J. We propose LoRA-adapted LLaMA-2-7B for NER. "
        "Training used rank-8 adapters on CoNLL-2003 and WikiANN (7 languages). "
        "Methodology: QLoRA fine-tuning with 4-bit NF4 quantization, gradient checkpointing. "
        "Cross-lingual transfer from English achieves 61.2% F1 macro-average. "
        "Wilcoxon signed-rank test confirms significance (p=0.002). "
        "Limitation: Low-resource languages underperform by 15-20 F1 points."
    ),
    (
        "This work presents a GAN-based data augmentation pipeline for medical imaging. "
        "Training used Adam optimizer (lr=2e-4, beta1=0.5) on CheXpert and MIMIC-CXR. "
        "Authors: Park S, Nguyen T. FID score improved from 18.3 to 12.7. "
        "Key finding: Augmentation reduced false negative rate by 23% on pneumonia detection. "
        "Bootstrap significance test (n=1000, CI 95%: 19-27%). "
        "Limitation: Trained only on frontal-view chest X-rays."
    ),
]


def _sample_text(rng: random.Random) -> str:
    return rng.choice(SAMPLE_TEXTS)


async def _single_request(
    client: httpx.AsyncClient,
    base_url: str,
    text: str,
    api_key: str,
) -> float | None:
    """Return request latency in seconds, or None on error."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{base_url}/api/extract",
            json={"section_text": text, "max_tokens": 512},
            headers=headers,
            timeout=90.0,
        )
        elapsed = time.perf_counter() - t0
        if resp.status_code == 200:
            return elapsed
        return None
    except Exception:
        return None


async def run_benchmark(
    n: int,
    concurrency: int,
    base_url: str,
    api_key: str,
    seed: int,
    warmup: int,
    label: str,
    output_path: Path | None,
) -> LatencyStats:
    rng = random.Random(seed)

    async with httpx.AsyncClient() as client:
        # Reachability check
        try:
            r = await client.get(f"{base_url}/health", timeout=5.0)
            if r.status_code != 200:
                raise ConnectionError(f"health check failed: HTTP {r.status_code}")
        except httpx.ConnectError:
            print(f"✗ Cannot reach {base_url}. Start with: docker compose up")
            raise SystemExit(1)

        # Warm-up (not counted in results)
        if warmup > 0:
            print(f"Warm-up: {warmup} requests...")
            warmup_texts = [_sample_text(rng) for _ in range(warmup)]
            semaphore = asyncio.Semaphore(concurrency)
            async def _warmup(text):
                async with semaphore:
                    await _single_request(client, base_url, text, api_key)
            await asyncio.gather(*[_warmup(t) for t in warmup_texts])

        # Main benchmark
        texts = [_sample_text(rng) for _ in range(n)]
        semaphore = asyncio.Semaphore(concurrency)
        latencies: list[float] = []
        errors = 0
        completed = 0

        t_total_start = time.perf_counter()

        async def _measure(text: str) -> None:
            nonlocal errors, completed
            async with semaphore:
                lat = await _single_request(client, base_url, text, api_key)
                if lat is None:
                    errors += 1
                else:
                    latencies.append(lat)
                completed += 1
                if completed % max(1, n // 10) == 0:
                    pct = completed / n * 100
                    print(f"  {completed}/{n} ({pct:.0f}%)  errors={errors}")

        print(f"\nBenchmark: {n} requests, concurrency={concurrency}")
        await asyncio.gather(*[_measure(t) for t in texts])
        t_total = time.perf_counter() - t_total_start

    if not latencies:
        print("✗ All requests failed. Check the API logs.")
        raise SystemExit(1)

    stats = LatencyStats.from_samples(latencies, concurrency=concurrency)
    error_rate = errors / n

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*54}")
    print(f"  {label}")
    print(f"{'='*54}")
    print(f"  Requests:     {n}  (errors: {errors}, {error_rate:.1%})")
    print(f"  Concurrency:  {concurrency}")
    print(f"  Wall time:    {t_total:.1f}s")
    print(f"  Throughput:   {stats.throughput_rps} req/s (Little's Law)")
    print(f"  Wall-clock:   {n / t_total:.2f} req/s (actual)")
    print(f"  Latency p50:  {stats.p50_s:.3f}s")
    print(f"  Latency p90:  {stats.p90_s:.3f}s")
    print(f"  Latency p95:  {stats.p95_s:.3f}s")
    print(f"  Latency p99:  {stats.p99_s:.3f}s")
    print(f"  Latency max:  {stats.max_s:.3f}s")
    print(f"  Latency mean: {stats.mean_s:.3f}s ± {stats.std_s:.3f}s")
    print(f"{'='*54}\n")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "label": label,
            "n": n,
            "errors": errors,
            "error_rate": error_rate,
            "concurrency": concurrency,
            "wall_time_s": round(t_total, 3),
            "actual_throughput_rps": round(n / t_total, 2),
            **{k: getattr(stats, k) for k in [
                "mean_s", "p50_s", "p90_s", "p95_s", "p99_s",
                "max_s", "min_s", "std_s", "throughput_rps",
            ]},
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {output_path}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests")
    parser.add_argument("--url", default="http://localhost:8080", help="API base URL")
    parser.add_argument("--api-key", default="", help="Bearer token")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up requests (not counted)")
    parser.add_argument("--label", default="benchmark", help="Label for this run")
    parser.add_argument("--output", type=Path, default=None, help="Save JSON results here")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        n=args.n,
        concurrency=args.concurrency,
        base_url=args.url,
        api_key=args.api_key,
        seed=args.seed,
        warmup=args.warmup,
        label=args.label,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
