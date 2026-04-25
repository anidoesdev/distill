"""Integration tests for FastAPI endpoints.

Uses httpx.AsyncClient with ASGITransport to run the app in-process —
no server needed, real HTTP semantics, lifespan fires normally.

VLLMClient is patched at the point of use so tests are hermetic and fast.
Async tests use @pytest.mark.anyio (anyio is available; pytest-asyncio is not).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from extractor.api.main import app
from tests.conftest import VALID_JSON_RESPONSE, VALID_META, make_mock_vllm_client


def _transport():
    return ASGITransport(app=app)


# ── /health ───────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_health_returns_200():
    async with AsyncClient(transport=_transport(), base_url="http://test") as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_health_no_auth_required():
    async with AsyncClient(transport=_transport(), base_url="http://test") as c:
        resp = await c.get("/health")
    assert resp.status_code == 200


# ── /api/info ─────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_info_returns_model_fields():
    mock_client = make_mock_vllm_client()
    with patch("extractor.api.main.VLLMClient", return_value=mock_client):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get("/api/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "model" in body
    assert "vllm_healthy" in body
    assert "max_retries" in body
    assert "auth_enabled" in body


@pytest.mark.anyio
async def test_info_vllm_unhealthy():
    mock_client = make_mock_vllm_client(health=False)
    with patch("extractor.api.main.VLLMClient", return_value=mock_client):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.get("/api/info")
    assert resp.status_code == 200
    assert resp.json()["vllm_healthy"] is False


# ── /api/extract ──────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_extract_success():
    mock_client = make_mock_vllm_client()
    with patch("extractor.api.main.VLLMClient", return_value=mock_client):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/extract",
                json={"section_text": "We trained a CNN on ImageNet using SGD." * 3},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert "extraction" in body
    assert set(body["extraction"].keys()) == {
        "authors", "methodology", "datasets_used",
        "key_findings", "limitations", "statistical_tests",
    }
    assert body["parse_error"] is None


@pytest.mark.anyio
async def test_extract_response_has_all_meta_fields():
    mock_client = make_mock_vllm_client()
    with patch("extractor.api.main.VLLMClient", return_value=mock_client):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/extract",
                json={"section_text": "Authors: Smith J. We trained a CNN." * 3},
            )
    body = resp.json()
    for field in ["latency_s", "prompt_tokens", "completion_tokens",
                  "repair_attempted", "repair_attempts"]:
        assert field in body, f"missing field: {field}"


@pytest.mark.anyio
async def test_extract_section_too_short_rejected():
    async with AsyncClient(transport=_transport(), base_url="http://test") as c:
        resp = await c.post("/api/extract", json={"section_text": "short"})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_extract_section_too_long_rejected():
    async with AsyncClient(transport=_transport(), base_url="http://test") as c:
        resp = await c.post("/api/extract", json={"section_text": "x" * 8001})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_extract_missing_body_rejected():
    async with AsyncClient(transport=_transport(), base_url="http://test") as c:
        resp = await c.post("/api/extract", json={})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_extract_with_repair():
    """When vLLM returns broken JSON, repair runs and result is still returned."""
    broken_then_ok_client = MagicMock()
    broken_then_ok_client.__aenter__ = AsyncMock(return_value=broken_then_ok_client)
    broken_then_ok_client.__aexit__ = AsyncMock(return_value=False)
    broken_then_ok_client.health = AsyncMock(return_value=True)
    broken_then_ok_client.list_models = AsyncMock(return_value=["extractor"])
    broken_then_ok_client.chat = AsyncMock(side_effect=[
        ("not json at all", dict(VALID_META)),
        (VALID_JSON_RESPONSE, dict(VALID_META)),
    ])

    with patch("extractor.api.main.VLLMClient", return_value=broken_then_ok_client):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/extract",
                json={"section_text": "Authors: Smith. We trained a model. " * 5},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["repair_attempted"] is True
    assert body["parse_error"] is None


# ── /api/extract/batch ────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_batch_extract_success():
    mock_client = make_mock_vllm_client()
    with patch("extractor.api.batch.VLLMClient", return_value=mock_client):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/extract/batch",
                json={"sections": [
                    "We trained CNN on ImageNet. Authors: Smith J. Key finding: 73% accuracy." * 2,
                    "Authors: Lee K. Used BERT on MIMIC-III. Limitation: English only." * 2,
                ]},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n"] == 2
    assert len(body["results"]) == 2
    assert "latency_s" in body


@pytest.mark.anyio
async def test_batch_extract_empty_list_rejected():
    async with AsyncClient(transport=_transport(), base_url="http://test") as c:
        resp = await c.post("/api/extract/batch", json={"sections": []})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_batch_extract_too_many_sections_rejected():
    async with AsyncClient(transport=_transport(), base_url="http://test") as c:
        resp = await c.post(
            "/api/extract/batch",
            json={"sections": ["section text " * 5] * 21},
        )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_batch_results_order_preserved():
    """Results must be in the same order as input sections."""
    authors = ["Author A", "Author B", "Author C"]
    call_count = 0

    def make_author_response(author: str) -> str:
        return json.dumps({
            "authors": [author], "methodology": "CNN", "datasets_used": [],
            "key_findings": [], "limitations": [], "statistical_tests": [],
        })

    async def chat_side_effect(messages, **kwargs):
        nonlocal call_count
        resp = make_author_response(authors[call_count])
        call_count += 1
        return resp, dict(VALID_META)

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.chat = AsyncMock(side_effect=chat_side_effect)

    with patch("extractor.api.batch.VLLMClient", return_value=mock_client):
        async with AsyncClient(transport=_transport(), base_url="http://test") as c:
            resp = await c.post(
                "/api/extract/batch",
                json={"sections": [
                    "Section one text here." * 3,
                    "Section two text here." * 3,
                    "Section three text here." * 3,
                ]},
            )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["extraction"]["authors"] == ["Author A"]
    assert results[1]["extraction"]["authors"] == ["Author B"]
    assert results[2]["extraction"]["authors"] == ["Author C"]
