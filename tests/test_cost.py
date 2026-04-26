"""Unit tests for extractor.utils.cost — cost and throughput calculations."""

import pytest

from extractor.utils.cost import (
    LatencyStats,
    ModelVariant,
    break_even_requests,
    cost_per_1k,
    estimate_costs,
    format_cost_table,
    throughput_at_concurrency,
)


# ── cost_per_1k ───────────────────────────────────────────────────────────────

def test_cost_per_1k_basic():
    # 1000 requests × 1.0s each = 1000s = 0.2778 GPU-hours × $3.50 = $0.9722
    result = cost_per_1k(mean_latency_s=1.0, gpu_price_per_hr=3.50)
    assert abs(result - (1000 * 1.0 / 3600 * 3.50)) < 1e-2


def test_cost_per_1k_awq_cheaper_than_bf16():
    bf16 = cost_per_1k(mean_latency_s=1.05, gpu_price_per_hr=3.50)
    awq  = cost_per_1k(mean_latency_s=0.30, gpu_price_per_hr=3.50)
    assert awq < bf16


def test_cost_per_1k_proportional_to_latency():
    c1 = cost_per_1k(mean_latency_s=1.0, gpu_price_per_hr=1.0)
    c2 = cost_per_1k(mean_latency_s=2.0, gpu_price_per_hr=1.0)
    assert abs(c2 / c1 - 2.0) < 1e-9


def test_cost_per_1k_proportional_to_gpu_price():
    c1 = cost_per_1k(mean_latency_s=1.0, gpu_price_per_hr=1.0)
    c2 = cost_per_1k(mean_latency_s=1.0, gpu_price_per_hr=3.5)
    # 4dp rounding means ratio is approximate
    assert abs(c2 / c1 - 3.5) < 0.01


# ── throughput_at_concurrency ─────────────────────────────────────────────────

def test_throughput_concurrency_1():
    tput = throughput_at_concurrency(mean_latency_s=1.0, concurrency=1)
    assert tput == 1.0


def test_throughput_scales_with_concurrency():
    t1 = throughput_at_concurrency(mean_latency_s=1.0, concurrency=1)
    t4 = throughput_at_concurrency(mean_latency_s=1.0, concurrency=4)
    assert abs(t4 / t1 - 4.0) < 1e-6


def test_throughput_zero_latency_returns_zero():
    assert throughput_at_concurrency(mean_latency_s=0.0, concurrency=4) == 0.0


# ── break_even_requests ───────────────────────────────────────────────────────

def test_break_even_positive():
    n = break_even_requests(
        baseline_latency_s=1.0,
        optimized_latency_s=0.3,
        gpu_price_per_hr=3.50,
        optimization_cost_gpu_hrs=2.0,
    )
    assert n > 0


def test_break_even_no_improvement_returns_minus_one():
    n = break_even_requests(
        baseline_latency_s=0.3,   # optimized is slower
        optimized_latency_s=1.0,
        gpu_price_per_hr=3.50,
        optimization_cost_gpu_hrs=2.0,
    )
    assert n == -1


def test_break_even_larger_savings_breaks_even_sooner():
    n_small = break_even_requests(1.0, 0.9, 3.50, 2.0)  # small saving
    n_large = break_even_requests(1.0, 0.3, 3.50, 2.0)  # large saving
    assert n_large < n_small


def test_break_even_awq_realistic():
    """AWQ (2 GPU-hrs to calibrate) should break even well under 100K requests."""
    n = break_even_requests(
        baseline_latency_s=1.05,
        optimized_latency_s=0.30,
        gpu_price_per_hr=3.50,
        optimization_cost_gpu_hrs=2.0,
    )
    assert 0 < n < 100_000


# ── LatencyStats ──────────────────────────────────────────────────────────────

def test_latency_stats_single_sample():
    stats = LatencyStats.from_samples([1.0], concurrency=1)
    assert stats.mean_s == 1.0
    assert stats.p50_s == 1.0
    assert stats.p99_s == 1.0
    assert stats.n == 1
    assert stats.throughput_rps == 1.0


def test_latency_stats_sorted_correctly():
    stats = LatencyStats.from_samples([3.0, 1.0, 2.0], concurrency=1)
    assert stats.min_s == 1.0
    assert stats.max_s == 3.0


def test_latency_stats_empty_raises():
    with pytest.raises(ValueError):
        LatencyStats.from_samples([], concurrency=1)


def test_latency_stats_throughput_scales_with_concurrency():
    stats1 = LatencyStats.from_samples([1.0] * 10, concurrency=1)
    stats4 = LatencyStats.from_samples([1.0] * 10, concurrency=4)
    assert stats4.throughput_rps == 4 * stats1.throughput_rps


def test_latency_stats_percentiles_ordered():
    samples = list(range(1, 101))  # 1..100 seconds
    stats = LatencyStats.from_samples([float(s) for s in samples], concurrency=1)
    assert stats.min_s <= stats.p50_s <= stats.p90_s <= stats.p95_s <= stats.p99_s <= stats.max_s


# ── estimate_costs ────────────────────────────────────────────────────────────

def test_estimate_costs_returns_one_per_variant():
    variants = [
        ModelVariant("BF16", 1.0, 3.5),
        ModelVariant("AWQ",  0.3, 1.1),
    ]
    estimates = estimate_costs(variants, "A100", 3.50, 80.0)
    assert len(estimates) == 2


def test_estimate_costs_fits_on_gpu():
    variants = [
        ModelVariant("Small", 1.0, 1.1),   # fits on 24GB
        ModelVariant("Large", 1.0, 80.0),  # doesn't fit on 24GB
    ]
    estimates = estimate_costs(variants, "A10G", 1.50, 24.0)
    assert estimates[0].fits_on_gpu is True
    assert estimates[1].fits_on_gpu is False


def test_estimate_costs_awq_cheaper():
    variants = [
        ModelVariant("BF16", 1.05, 3.5),
        ModelVariant("AWQ",  0.30, 1.1),
    ]
    estimates = estimate_costs(variants, "A100", 3.50, 80.0)
    bf16_cost = next(e for e in estimates if e.variant_name == "BF16").cost_per_1k_usd
    awq_cost  = next(e for e in estimates if e.variant_name == "AWQ").cost_per_1k_usd
    assert awq_cost < bf16_cost


# ── format_cost_table ─────────────────────────────────────────────────────────

def test_format_cost_table_contains_headers():
    variants = [ModelVariant("BF16", 1.0, 3.5)]
    estimates = estimate_costs(variants, "A100", 3.50, 80.0)
    table = format_cost_table(estimates)
    assert "Variant" in table
    assert "Cost/1K" in table
    assert "BF16" in table


def test_format_cost_table_is_string():
    estimates = estimate_costs(
        [ModelVariant("AWQ", 0.3, 1.1)],
        "RTX 4090", 0.50, 24.0,
    )
    assert isinstance(format_cost_table(estimates), str)
