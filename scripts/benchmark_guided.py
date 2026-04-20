"""Benchmark parse failure rate: guided decoding vs. standard retry+repair.

Sends N requests to the running extractor API, alternating between:
  - /api/extract (standard path, use_guided_decoding=False)
  - /api/extract with USE_GUIDED_DECODING=1 (guided path, no parse failures)

Reports:
  - Parse failure rate for each path
  - Average latency per path
  - Repair attempt rate and mean repair count (standard path only)
  - Token throughput (tokens/sec)

Prerequisites:
    docker compose up        # start vLLM + extractor-api
    python scripts/benchmark_guided.py --n 50

To test guided decoding, restart the API with:
    USE_GUIDED_DECODING=1 uvicorn extractor.api.main:app --port 8081

Then pass --guided-url http://localhost:8081 to this script.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx


# Short synthetic paper snippets — varied to exercise the model
SAMPLE_TEXTS = [
    (
        "We evaluate BERT-Large on SQuAD v1.1 and v2.0 using the Adam optimizer "
        "with lr=3e-5, batch size 32, and 3 epochs. Authors: Devlin et al. (2019). "
        "Our method achieves 93.2% F1 on SQuAD v1.1, exceeding the human baseline "
        "of 91.2%. Limitations include high compute cost and English-only training data."
    ),
    (
        "Authors: Smith J, Lee K, Patel R. We introduce a transformer variant trained "
        "on PubMed abstracts and MIMIC-III clinical notes. SGD with momentum 0.9 was "
        "used. The model reduces medication error detection time by 40% (p<0.001, t-test). "
        "Key limitations: dataset imbalance and limited to English clinical notes."
    ),
    (
        "This work presents a GAN-based data augmentation pipeline. Training used Adam "
        "optimizer (lr=2e-4, beta1=0.5) on CIFAR-10 and STL-10. FID score improved "
        "from 18.3 to 12.7. Statistical significance tested via bootstrap (n=1000). "
        "No author information provided. Primary limitation: mode collapse on rare classes."
    ),
    (
        "We propose LoRA-adapted LLaMA-2-7B for code completion tasks. "
        "Fine-tuning used rank-8 adapters on CodeSearchNet (Python subset). "
        "Authors: Zhang Wei, Hernandez M, Okonkwo C. Pass@1 improved from 23.1% to "
        "41.8% vs. baseline. Mann-Whitney U test (p=0.003). Limitation: single language."
    ),
    (
        "Cross-lingual transfer with mBERT across 7 NER datasets. Zero-shot transfer "
        "from English achieves 61.2% F1 macro-average. Authors: Müller A, Santos B. "
        "Datasets: CoNLL-2003, WikiANN, MultiNERD. Wilcoxon signed-rank test confirms "
        "significance. Limitation: low-resource languages underperform by 15-20 F1 points."
    ),
]


def _sample_text(rng: random.Random) -> str:
    return rng.choice(SAMPLE_TEXTS)


async def _send_request(
    client: httpx.AsyncClient,
    base_url: str,
    text: str,
    api_key: str,
) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{base_url}/api/extract",
            json={"section_text": text, "max_tokens": 512},
            headers=headers,
            timeout=60.0,
        )
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        body = resp.json()
        body["_request_latency_s"] = round(elapsed, 3)
        body["_http_error"] = None
        return body
    except httpx.HTTPStatusError as e:
        return {"_http_error": f"HTTP {e.response.status_code}", "_request_latency_s": 0.0}
    except Exception as e:
        return {"_http_error": str(e), "_request_latency_s": 0.0}


def _aggregate(results: list[dict], label: str) -> dict:
    total = len(results)
    errors = sum(1 for r in results if r.get("_http_error"))
    ok = [r for r in results if not r.get("_http_error")]

    parse_failures = sum(1 for r in ok if r.get("parse_error"))
    repair_attempted = sum(1 for r in ok if r.get("repair_attempted"))
    repair_counts = [r.get("repair_attempts", 0) for r in ok if r.get("repair_attempted")]
    latencies = [r["_request_latency_s"] for r in ok]
    completion_tokens = [r.get("completion_tokens", 0) for r in ok]
    latency_s_server = [r.get("latency_s", 0.0) for r in ok]

    return {
        "label": label,
        "n": total,
        "http_errors": errors,
        "ok_requests": len(ok),
        "parse_failure_rate": parse_failures / len(ok) if ok else 0.0,
        "parse_failures": parse_failures,
        "repair_rate": repair_attempted / len(ok) if ok else 0.0,
        "mean_repair_attempts": sum(repair_counts) / len(repair_counts) if repair_counts else 0.0,
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "p99_latency_s": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0.0,
        "mean_server_latency_s": sum(latency_s_server) / len(latency_s_server) if latency_s_server else 0.0,
        "mean_completion_tokens": sum(completion_tokens) / len(completion_tokens) if completion_tokens else 0.0,
    }


def _print_report(standard: dict, guided: dict) -> None:
    print("\n" + "=" * 62)
    print("  Guided Decoding Benchmark Report")
    print("=" * 62)

    rows = [
        ("Requests sent",         f"{standard['n']}",                 f"{guided['n']}"),
        ("HTTP errors",           f"{standard['http_errors']}",        f"{guided['http_errors']}"),
        ("Parse failures",        f"{standard['parse_failures']}",     f"{guided['parse_failures']}"),
        ("Parse failure rate",    f"{standard['parse_failure_rate']:.1%}", f"{guided['parse_failure_rate']:.1%}"),
        ("Repair rate",           f"{standard['repair_rate']:.1%}",    "n/a"),
        ("Mean repair attempts",  f"{standard['mean_repair_attempts']:.2f}", "n/a"),
        ("Mean latency (client)", f"{standard['mean_latency_s']:.3f}s", f"{guided['mean_latency_s']:.3f}s"),
        ("p99 latency (client)",  f"{standard['p99_latency_s']:.3f}s",  f"{guided['p99_latency_s']:.3f}s"),
        ("Mean server latency",   f"{standard['mean_server_latency_s']:.3f}s", f"{guided['mean_server_latency_s']:.3f}s"),
    ]

    header = f"{'Metric':<28} {'Standard':>14} {'Guided':>14}"
    print(header)
    print("-" * 62)
    for label, std, gd in rows:
        print(f"{label:<28} {std:>14} {gd:>14}")

    print("=" * 62)

    # Summary verdict
    pfr_std = standard["parse_failure_rate"]
    pfr_gd = guided["parse_failure_rate"]
    lat_std = standard["mean_server_latency_s"]
    lat_gd = guided["mean_server_latency_s"]
    lat_overhead = (lat_gd - lat_std) / lat_std * 100 if lat_std > 0 else 0.0

    print(f"\nVerdict:")
    if pfr_gd == 0.0:
        print("  ✓ Guided decoding eliminated all parse failures.")
    elif pfr_gd < pfr_std:
        print(f"  ↓ Parse failures reduced: {pfr_std:.1%} → {pfr_gd:.1%}")
    else:
        print(f"  ✗ Guided decoding did not reduce parse failures.")

    if lat_overhead > 5:
        print(f"  ⚠ Latency overhead: +{lat_overhead:.1f}% (constrained decoding is slower).")
    elif lat_overhead < 0:
        print(f"  ✓ Guided decoding is {-lat_overhead:.1f}% faster (no repair round-trips).")
    else:
        print(f"  ~ Latency overhead: {lat_overhead:+.1f}% (within noise).")
    print()


async def run_benchmark(
    n: int,
    standard_url: str,
    guided_url: str,
    api_key: str,
    seed: int,
    output_path: Path | None,
) -> None:
    rng = random.Random(seed)
    texts = [_sample_text(rng) for _ in range(n)]

    print(f"Sending {n} requests to standard API  ({standard_url}) ...")
    async with httpx.AsyncClient() as client:
        standard_results = []
        for i, text in enumerate(texts, 1):
            r = await _send_request(client, standard_url, text, api_key)
            standard_results.append(r)
            if i % 10 == 0:
                print(f"  {i}/{n} done")

    print(f"Sending {n} requests to guided API  ({guided_url}) ...")
    async with httpx.AsyncClient() as client:
        guided_results = []
        for i, text in enumerate(texts, 1):
            r = await _send_request(client, guided_url, text, api_key)
            guided_results.append(r)
            if i % 10 == 0:
                print(f"  {i}/{n} done")

    standard_agg = _aggregate(standard_results, "standard")
    guided_agg = _aggregate(guided_results, "guided")

    _print_report(standard_agg, guided_agg)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {"standard": standard_agg, "guided": guided_agg},
                f,
                indent=2,
            )
        print(f"Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="Number of requests per path (default: 50)")
    parser.add_argument("--standard-url", default="http://localhost:8080", help="Standard API base URL")
    parser.add_argument("--guided-url", default="http://localhost:8081", help="Guided API base URL")
    parser.add_argument("--api-key", default="", help="Bearer token (leave empty if auth disabled)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON results to this path (optional)")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        n=args.n,
        standard_url=args.standard_url,
        guided_url=args.guided_url,
        api_key=args.api_key,
        seed=args.seed,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
