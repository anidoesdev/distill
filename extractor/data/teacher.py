"""Async teacher model client for dataset distillation.

Supports OpenAI (gpt-4o-mini) and Anthropic (claude-haiku-4-5) as teacher models.
The caller sees one interface; the provider is selected at construction time.

Usage:
    async with TeacherClient("openai") as client:
        result, error, usage = await client.extract(section_text)
"""

from __future__ import annotations

import asyncio
import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from tenacity import retry, stop_after_attempt, wait_exponential

from extractor.prompt import build_messages
from extractor.schemas.extraction import ExtractionResult
from extractor.utils.logging import get_logger

logger = get_logger(__name__)

Provider = Literal["openai", "anthropic"]

# Model IDs for each provider — pinned so cost estimates stay accurate
TEACHER_MODELS: dict[Provider, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}

# $/1M tokens (approximate, March 2026)
COST_PER_1M: dict[Provider, dict[str, float]] = {
    "openai":    {"input": 0.15, "output": 0.60},
    "anthropic": {"input": 0.25, "output": 1.25},
}


class UsageAccumulator:
    """Thread-safe running token usage tracker."""

    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        self._input_tokens = 0
        self._output_tokens = 0
        self._lock = asyncio.Lock()

    async def add(self, input_tokens: int, output_tokens: int) -> None:
        async with self._lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens

    def cost_usd(self) -> float:
        rates = COST_PER_1M[self.provider]
        return (
            self._input_tokens / 1e6 * rates["input"]
            + self._output_tokens / 1e6 * rates["output"]
        )

    def summary(self) -> dict:
        return {
            "provider": self.provider,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "estimated_cost_usd": round(self.cost_usd(), 4),
        }


class TeacherClient:
    """Async teacher model client.

    Args:
        provider: "openai" or "anthropic"
        max_concurrency: Max simultaneous in-flight API calls. Set conservatively
            to stay under rate limits. OpenAI tier-1 default: 500 RPM → use 5-10.
    """

    def __init__(
        self,
        provider: Provider = "openai",
        max_concurrency: int = 5,
    ) -> None:
        self.provider = provider
        self.model = TEACHER_MODELS[provider]
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.usage = UsageAccumulator(provider)
        self._client: object | None = None

    async def __aenter__(self) -> TeacherClient:
        if self.provider == "openai":
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        logger.info(
            "teacher client ready",
            extra={"provider": self.provider, "model": self.model},
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        summary = self.usage.summary()
        logger.info("teacher session complete", extra=summary)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _call_api(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        """Call the provider API. Returns (response_text, input_tokens, output_tokens)."""
        if self.provider == "openai":
            from openai import AsyncOpenAI
            client = self._client  # type: ignore[assignment]
            resp = await client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=1024,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
            return content, resp.usage.prompt_tokens, resp.usage.completion_tokens

        else:  # anthropic
            from anthropic import AsyncAnthropic
            client = self._client  # type: ignore[assignment]
            # Anthropic API: system is a top-level param, not a message role
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_messages = [m for m in messages if m["role"] != "system"]
            resp = await client.messages.create(  # type: ignore[union-attr]
                model=self.model,
                system=system,
                messages=user_messages,  # type: ignore[arg-type]
                max_tokens=1024,
            )
            content = resp.content[0].text if resp.content else ""
            return content, resp.usage.input_tokens, resp.usage.output_tokens

    async def extract(
        self, section_text: str
    ) -> tuple[ExtractionResult, str | None, dict]:
        """Extract structured info from one paper section.

        Returns:
            (result, error, usage_dict)
            error is None on success. result is always a valid ExtractionResult.
        """
        async with self._semaphore:
            messages = build_messages(section_text)
            try:
                raw, n_in, n_out = await self._call_api(messages)
            except Exception as exc:
                return ExtractionResult(), f"APIError: {exc}", {}

            await self.usage.add(n_in, n_out)
            result, error = ExtractionResult.from_model_output(raw)

            usage = {"input_tokens": n_in, "output_tokens": n_out}
            return result, error, usage
