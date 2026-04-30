"""Prometheus metrics for the EXTRACTOR API.

Metrics are created lazily via prometheus_client. If the library is not
installed, all metric objects are replaced with no-op stubs so the API
starts cleanly without Prometheus.

Install: pip install prometheus-client

Usage:
    from extractor.utils.metrics import METRICS
    METRICS.requests_total.labels(endpoint="/api/extract", status="200").inc()
    METRICS.request_latency.labels(endpoint="/api/extract").observe(0.42)

Exposed at GET /metrics (added to main.py in this session).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ── No-op stubs (used when prometheus_client is not installed) ────────────────

class _NoOpCounter:
    def labels(self, **_: Any) -> "_NoOpCounter":
        return self
    def inc(self, amount: float = 1) -> None:
        pass


class _NoOpHistogram:
    def labels(self, **_: Any) -> "_NoOpHistogram":
        return self
    def observe(self, value: float) -> None:
        pass


class _NoOpGauge:
    def labels(self, **_: Any) -> "_NoOpGauge":
        return self
    def set(self, value: float) -> None:
        pass
    def inc(self, amount: float = 1) -> None:
        pass
    def dec(self, amount: float = 1) -> None:
        pass


# ── Metric definitions ────────────────────────────────────────────────────────

# Latency buckets matched to our SLO: p50 ~0.3s, p99 target <2s
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf"))


@dataclass
class ExtractorMetrics:
    """Container for all Prometheus metric objects."""

    requests_total: Any          # Counter{endpoint, status}
    request_latency: Any         # Histogram{endpoint}
    parse_failures_total: Any    # Counter{}
    repair_attempts_total: Any   # Counter{}
    vllm_latency: Any            # Histogram{}
    active_requests: Any         # Gauge{}
    prometheus_available: bool


def _build_metrics() -> ExtractorMetrics:
    try:
        from prometheus_client import Counter, Gauge, Histogram

        return ExtractorMetrics(
            requests_total=Counter(
                "extractor_requests_total",
                "Total HTTP requests processed",
                ["endpoint", "status"],
            ),
            request_latency=Histogram(
                "extractor_request_latency_seconds",
                "End-to-end request latency in seconds",
                ["endpoint"],
                buckets=_LATENCY_BUCKETS,
            ),
            parse_failures_total=Counter(
                "extractor_parse_failures_total",
                "Number of model outputs that failed JSON parsing",
            ),
            repair_attempts_total=Counter(
                "extractor_repair_attempts_total",
                "Number of parse-repair attempts made (each costs one model call)",
            ),
            vllm_latency=Histogram(
                "extractor_vllm_latency_seconds",
                "vLLM inference latency (server-side, from meta)",
                buckets=_LATENCY_BUCKETS,
            ),
            active_requests=Gauge(
                "extractor_active_requests",
                "Number of requests currently being processed",
                ["endpoint"],
            ),
            prometheus_available=True,
        )
    except ImportError:
        return ExtractorMetrics(
            requests_total=_NoOpCounter(),
            request_latency=_NoOpHistogram(),
            parse_failures_total=_NoOpCounter(),
            repair_attempts_total=_NoOpCounter(),
            vllm_latency=_NoOpHistogram(),
            active_requests=_NoOpGauge(),
            prometheus_available=False,
        )


# Module-level singleton — created once at import time.
METRICS = _build_metrics()
