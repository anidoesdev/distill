"""Cost and throughput calculation utilities for ML serving analysis.

All functions are pure (no I/O) so they're testable and importable
without any infrastructure running.

GPU price reference (on-demand, 2025):
  A100 80GB:  $3.50/hr (Lambda), $2.21/hr (vast.ai spot)
  A10G 24GB:  $1.50/hr (AWS g5), $0.75/hr (vast.ai spot)
  RTX 4090:   $0.50/hr (vast.ai spot)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LatencyStats:
    """Aggregated latency statistics from a benchmark run."""

    n: int
    mean_s: float
    p50_s: float
    p90_s: float
    p95_s: float
    p99_s: float
    max_s: float
    min_s: float
    std_s: float
    throughput_rps: float  # requests per second at measured concurrency
    concurrency: int

    @classmethod
    def from_samples(cls, samples: list[float], concurrency: int = 1) -> "LatencyStats":
        """Compute stats from a list of per-request latency values (seconds)."""
        if not samples:
            raise ValueError("No samples provided")
        n = len(samples)
        sorted_s = sorted(samples)
        mean_s = sum(sorted_s) / n
        std_s = (sum((x - mean_s) ** 2 for x in sorted_s) / n) ** 0.5
        throughput_rps = concurrency / mean_s if mean_s > 0 else 0.0

        def pct(p: float) -> float:
            idx = int(p / 100 * n)
            return sorted_s[min(idx, n - 1)]

        return cls(
            n=n,
            mean_s=round(mean_s, 4),
            p50_s=round(pct(50), 4),
            p90_s=round(pct(90), 4),
            p95_s=round(pct(95), 4),
            p99_s=round(pct(99), 4),
            max_s=round(sorted_s[-1], 4),
            min_s=round(sorted_s[0], 4),
            std_s=round(std_s, 4),
            throughput_rps=round(throughput_rps, 2),
            concurrency=concurrency,
        )


@dataclass
class ModelVariant:
    """Specification for one model variant in the cost comparison."""

    name: str
    mean_latency_s: float
    vram_gb: float
    quantization: str = "bf16"  # bf16, int4-awq, q4_k_m
    notes: str = ""


@dataclass
class CostEstimate:
    """Cost breakdown for a single model variant."""

    variant_name: str
    mean_latency_s: float
    throughput_rps: float
    cost_per_1k_usd: float
    gpu_name: str
    gpu_price_per_hr: float
    vram_required_gb: float
    fits_on_gpu: bool


def cost_per_1k(
    mean_latency_s: float,
    gpu_price_per_hr: float,
) -> float:
    """Compute USD cost to run 1000 extractions at steady-state throughput.

    Assumes one GPU serving one request at a time (concurrency=1).
    With batching, divide by concurrency to get the parallel-adjusted cost.

    Args:
        mean_latency_s: Average request duration in seconds.
        gpu_price_per_hr: GPU rental price in USD per hour.

    Returns:
        USD cost per 1000 extractions.
    """
    hours_per_1k = (1000 * mean_latency_s) / 3600
    return round(hours_per_1k * gpu_price_per_hr, 4)


def throughput_at_concurrency(mean_latency_s: float, concurrency: int) -> float:
    """Theoretical throughput (req/s) under Little's Law.

    Valid while the server is not saturated (queuing latency negligible).
    """
    if mean_latency_s <= 0:
        return 0.0
    return round(concurrency / mean_latency_s, 2)


def break_even_requests(
    baseline_latency_s: float,
    optimized_latency_s: float,
    gpu_price_per_hr: float,
    optimization_cost_gpu_hrs: float,
) -> int:
    """How many requests until an optimization (e.g., AWQ) pays for itself.

    Args:
        baseline_latency_s:    Mean latency before optimization.
        optimized_latency_s:   Mean latency after optimization.
        gpu_price_per_hr:      GPU rental price per hour.
        optimization_cost_gpu_hrs: One-time cost of the optimization run
                                   (e.g., 2 hrs for AWQ calibration).

    Returns:
        Number of requests at which cumulative savings equals optimization cost.
    """
    savings_per_request_usd = (
        (baseline_latency_s - optimized_latency_s) / 3600 * gpu_price_per_hr
    )
    if savings_per_request_usd <= 0:
        return -1  # no break-even (optimization is slower)
    one_time_cost = optimization_cost_gpu_hrs * gpu_price_per_hr
    return int(one_time_cost / savings_per_request_usd) + 1


def estimate_costs(
    variants: list[ModelVariant],
    gpu_name: str,
    gpu_price_per_hr: float,
    gpu_vram_gb: float,
) -> list[CostEstimate]:
    """Compute cost estimates for all model variants on a given GPU."""
    estimates = []
    for v in variants:
        tput = throughput_at_concurrency(v.mean_latency_s, concurrency=1)
        c1k = cost_per_1k(v.mean_latency_s, gpu_price_per_hr)
        fits = v.vram_gb <= gpu_vram_gb
        estimates.append(CostEstimate(
            variant_name=v.name,
            mean_latency_s=v.mean_latency_s,
            throughput_rps=tput,
            cost_per_1k_usd=c1k,
            gpu_name=gpu_name,
            gpu_price_per_hr=gpu_price_per_hr,
            vram_required_gb=v.vram_gb,
            fits_on_gpu=fits,
        ))
    return estimates


def format_cost_table(estimates: list[CostEstimate]) -> str:
    """Render a markdown cost comparison table."""
    header = (
        "| Variant | Latency (s) | Throughput | Cost/1K | VRAM (GB) | Fits GPU |\n"
        "|---------|------------|------------|---------|-----------|----------|\n"
    )
    rows = []
    for e in estimates:
        fits = "✓" if e.fits_on_gpu else "✗"
        rows.append(
            f"| {e.variant_name} | {e.mean_latency_s:.3f} | "
            f"{e.throughput_rps:.1f} req/s | ${e.cost_per_1k_usd:.4f} | "
            f"{e.vram_required_gb:.1f} | {fits} |"
        )
    return header + "\n".join(rows)
