"""Unit tests for extractor.api.repair — retry+repair logic.

All tests mock VLLMClient.chat so no network or model is needed.
Async tests use @pytest.mark.anyio (anyio is available, pytest-asyncio is not).
"""

from __future__ import annotations

import pytest

from extractor.api.repair import (
    REPAIR_SYSTEM_PROMPT,
    build_repair_messages,
    extract_with_retry,
    repair_and_parse,
)
from extractor.schemas.extraction import ExtractionResult


# ── build_repair_messages ─────────────────────────────────────────────────────

def test_build_repair_messages_structure():
    msgs = build_repair_messages("broken output here")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "broken output here" in msgs[1]["content"]


def test_build_repair_messages_system_prompt():
    msgs = build_repair_messages("x")
    assert msgs[0]["content"] == REPAIR_SYSTEM_PROMPT


def test_build_repair_messages_truncates_long_output():
    long_output = "x" * 5000
    msgs = build_repair_messages(long_output)
    # User content should contain at most 2000 chars of the broken output
    assert len(msgs[1]["content"]) < 2200  # 2000 chars + prompt prefix overhead


def test_build_repair_messages_short_output_not_truncated():
    output = "short broken json"
    msgs = build_repair_messages(output)
    assert output in msgs[1]["content"]


# ── repair_and_parse ──────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_repair_and_parse_success(mock_vllm_ok, valid_json_response):
    result, error = await repair_and_parse("broken", mock_vllm_ok)
    assert error is None
    assert not result.is_empty()
    assert result.authors == ["Alice Smith", "Bob Jones"]


@pytest.mark.anyio
async def test_repair_and_parse_returns_empty_on_failure(mock_vllm_always_broken):
    result, error = await repair_and_parse("broken", mock_vllm_always_broken)
    assert error is not None
    assert result.is_empty()


@pytest.mark.anyio
async def test_repair_and_parse_exception_handling():
    """If client.chat raises, repair returns empty result + error string."""
    from unittest.mock import AsyncMock, MagicMock
    client = MagicMock()
    client.chat = AsyncMock(side_effect=ConnectionError("vLLM down"))
    result, error = await repair_and_parse("broken", client)
    assert error is not None
    assert "vLLM down" in error
    assert result.is_empty()


# ── extract_with_retry ────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_extract_with_retry_success_first_try(mock_vllm_ok):
    messages = [{"role": "user", "content": "extract"}]
    result, error, meta = await extract_with_retry(messages, mock_vllm_ok, max_retries=2)
    assert error is None
    assert not result.is_empty()
    # Only one chat call — no repair needed
    assert mock_vllm_ok.chat.call_count == 1
    assert meta.get("repair_attempted", False) is False


@pytest.mark.anyio
async def test_extract_with_retry_repairs_on_first_failure(mock_vllm_broken):
    messages = [{"role": "user", "content": "extract"}]
    result, error, meta = await extract_with_retry(messages, mock_vllm_broken, max_retries=2)
    assert error is None
    assert not result.is_empty()
    # Two chat calls: initial (broken) + one repair (success)
    assert mock_vllm_broken.chat.call_count == 2
    assert meta["repair_attempted"] is True
    assert meta["repair_attempts"] == 1


@pytest.mark.anyio
async def test_extract_with_retry_exhausts_retries(mock_vllm_always_broken):
    messages = [{"role": "user", "content": "extract"}]
    result, error, meta = await extract_with_retry(
        messages, mock_vllm_always_broken, max_retries=2
    )
    assert error is not None
    assert result.is_empty()
    # 1 initial + 2 repair attempts = 3 total calls
    assert mock_vllm_always_broken.chat.call_count == 3
    assert meta["repair_attempts"] == 2


@pytest.mark.anyio
async def test_extract_with_retry_zero_retries(mock_vllm_always_broken):
    messages = [{"role": "user", "content": "extract"}]
    result, error, meta = await extract_with_retry(
        messages, mock_vllm_always_broken, max_retries=0
    )
    assert error is not None
    # Only 1 chat call — no repair attempts
    assert mock_vllm_always_broken.chat.call_count == 1


@pytest.mark.anyio
async def test_extract_with_retry_meta_contains_expected_keys(mock_vllm_ok):
    messages = [{"role": "user", "content": "extract"}]
    _, _, meta = await extract_with_retry(messages, mock_vllm_ok)
    assert "prompt_tokens" in meta
    assert "completion_tokens" in meta
    assert "latency_s" in meta
