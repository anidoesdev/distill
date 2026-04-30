"""Tests for health check endpoints and metrics module.

Health endpoint tests use httpx.AsyncClient + ASGITransport (in-process).
Metrics tests verify the no-op stub behaviour (prometheus_client may not be installed).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from extractor.api.main import app
from tests.conftest import make_mock_vllm_client


def _transport():
    return ASGITransport(app=app)


# ── /health (backwards-compatible alias) ──────────────────────────────────────

@pytest.mark.anyio
async def test_health_alias_returns_200():
    async with AsyncClient(transport=_transport(), base_url="http://test") as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── /health/live ──────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_liveness_returns_200():
    async with AsyncClient(transport=_transport(), base_url="http://test") as c:
        resp = await c.get("/health/live")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_liveness_body_has_status_and_uptime():
    async with AsyncClient(transport=_transport(), base_url="http://test") as c:
        resp = await c.get("/health/live")
    body = resp.json()
    assert body["status"] == "live"
    assert "uptime_s" in body
    assert isinstance(body["uptime_s"], float)


@pytest.mark.anyio
async def test_liveness_no_external_calls():
    """Liveness must not call vLLM — it should respond instantly even if vLLM is down."""
    with patch("extractor.api.health.VLLMClient") as mock_cls:
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get("/health/live")
    assert resp.status_code == 200
    mock_cls.assert_not_called()


# ── /health/ready ─────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_readiness_200_when_vllm_healthy():
    mock_client = make_mock_vllm_client(health=True)
    with patch("extractor.api.health.VLLMClient", return_value=mock_client):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["checks"]["vllm_reachable"] is True


@pytest.mark.anyio
async def test_readiness_503_when_vllm_unhealthy():
    mock_client = make_mock_vllm_client(health=False)
    with patch("extractor.api.health.VLLMClient", return_value=mock_client):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["vllm_reachable"] is False


@pytest.mark.anyio
async def test_readiness_503_on_vllm_exception():
    """If VLLMClient raises (e.g., ConnectionError), readiness returns 503."""
    error_client = MagicMock()
    error_client.__aenter__ = AsyncMock(side_effect=ConnectionError("refused"))
    error_client.__aexit__ = AsyncMock(return_value=False)
    with patch("extractor.api.health.VLLMClient", return_value=error_client):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get("/health/ready")
    assert resp.status_code == 503


@pytest.mark.anyio
async def test_readiness_body_has_model_and_uptime():
    mock_client = make_mock_vllm_client(health=True)
    with patch("extractor.api.health.VLLMClient", return_value=mock_client):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get("/health/ready")
    body = resp.json()
    assert "model" in body
    assert "uptime_s" in body


# ── /metrics endpoint ─────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_metrics_endpoint_exists():
    async with AsyncClient(transport=_transport(), base_url="http://test") as c:
        resp = await c.get("/metrics")
    # Either 200 (prometheus installed) or 503 (not installed) — never 404
    assert resp.status_code in (200, 503)


@pytest.mark.anyio
async def test_metrics_503_without_prometheus(monkeypatch):
    """When prometheus_client is not available, /metrics returns 503."""
    from extractor.utils import metrics as metrics_mod
    monkeypatch.setattr(metrics_mod.METRICS, "prometheus_available", False)
    async with AsyncClient(transport=_transport(), base_url="http://test") as c:
        resp = await c.get("/metrics")
    assert resp.status_code == 503


# ── Metrics module: no-op stubs ───────────────────────────────────────────────

def test_metrics_module_importable():
    from extractor.utils.metrics import METRICS, ExtractorMetrics  # noqa: F401


def test_metrics_noop_counter_does_not_raise():
    from extractor.utils.metrics import _NoOpCounter
    c = _NoOpCounter()
    c.labels(endpoint="/test", status="200").inc()
    c.labels(endpoint="/test", status="200").inc(5)


def test_metrics_noop_histogram_does_not_raise():
    from extractor.utils.metrics import _NoOpHistogram
    h = _NoOpHistogram()
    h.labels(endpoint="/test").observe(0.42)


def test_metrics_noop_gauge_does_not_raise():
    from extractor.utils.metrics import _NoOpGauge
    g = _NoOpGauge()
    g.labels(endpoint="/test").inc()
    g.labels(endpoint="/test").dec()
    g.labels(endpoint="/test").set(3.14)


def test_metrics_prometheus_available_is_bool():
    from extractor.utils.metrics import METRICS
    assert isinstance(METRICS.prometheus_available, bool)
