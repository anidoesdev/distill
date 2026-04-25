"""Shared pytest fixtures for all test modules."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from extractor.schemas.extraction import ExtractionResult

# Run anyio-marked tests with asyncio only (trio is not installed).
@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


# ── Canned model response ─────────────────────────────────────────────────────

VALID_JSON_RESPONSE = json.dumps({
    "authors": ["Alice Smith", "Bob Jones"],
    "methodology": "We trained a CNN on ImageNet using SGD with momentum 0.9.",
    "datasets_used": ["ImageNet", "CIFAR-10"],
    "key_findings": ["73.4% top-1 accuracy", "Outperforms baseline by 5.2%"],
    "limitations": ["English-only training data", "Not evaluated on out-of-distribution data"],
    "statistical_tests": ["paired t-test (p < 0.001)", "bootstrap CI (n=1000)"],
})

VALID_META = {
    "prompt_tokens": 120,
    "completion_tokens": 95,
    "total_tokens": 215,
    "finish_reason": "stop",
    "latency_s": 0.312,
    "tokens_per_sec": 304.5,
}


@pytest.fixture
def valid_json_response() -> str:
    return VALID_JSON_RESPONSE


@pytest.fixture
def valid_meta() -> dict:
    return dict(VALID_META)


@pytest.fixture
def sample_result() -> ExtractionResult:
    return ExtractionResult.model_validate(json.loads(VALID_JSON_RESPONSE))


@pytest.fixture
def empty_result() -> ExtractionResult:
    return ExtractionResult()


# ── Mock VLLMClient ───────────────────────────────────────────────────────────

def make_mock_vllm_client(
    response_text: str = VALID_JSON_RESPONSE,
    meta: dict | None = None,
    health: bool = True,
) -> MagicMock:
    """Return a mock VLLMClient instance that works as an async context manager."""
    mock_meta = dict(meta or VALID_META)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.chat = AsyncMock(return_value=(response_text, mock_meta))
    client.health = AsyncMock(return_value=health)
    client.list_models = AsyncMock(return_value=["extractor"])
    return client


@pytest.fixture
def mock_vllm_ok(valid_json_response, valid_meta):
    return make_mock_vllm_client(valid_json_response, valid_meta)


@pytest.fixture
def mock_vllm_broken():
    broken = "This is not JSON at all."
    repaired = VALID_JSON_RESPONSE
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.health = AsyncMock(return_value=True)
    client.list_models = AsyncMock(return_value=["extractor"])
    client.chat = AsyncMock(side_effect=[
        (broken, dict(VALID_META)),
        (repaired, dict(VALID_META)),
    ])
    return client


@pytest.fixture
def mock_vllm_always_broken():
    broken = "not json"
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.health = AsyncMock(return_value=True)
    client.list_models = AsyncMock(return_value=[])
    client.chat = AsyncMock(return_value=(broken, dict(VALID_META)))
    return client
