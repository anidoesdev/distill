"""vLLM inference client using the OpenAI-compatible HTTP API.

vLLM exposes /v1/chat/completions and /v1/completions endpoints that are
wire-compatible with the OpenAI SDK. This client wraps those endpoints with
the same interface as HFInference so the FastAPI layer doesn't need to know
which backend is in use.

vLLM must be running before calls are made:
    docker compose up vllm
    # or directly:
    vllm serve checkpoints/awq --quantization awq --max-model-len 1024 --port 8000

The client uses httpx for async HTTP so it integrates cleanly with the
FastAPI async request handler in session 23.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from extractor.config import settings
from extractor.utils.logging import get_logger

logger = get_logger(__name__)

# vLLM returns finish_reason="stop" on normal completion
_STOP_REASONS = {"stop", "eos"}


class VLLMClient:
    """Async client for a vLLM OpenAI-compatible server.

    Usage (within an async context):
        async with VLLMClient() as client:
            raw, meta = await client.chat(messages)
            result, error = ExtractionResult.from_model_output(raw)
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (base_url or settings.vllm_base_url).rstrip("/")
        self.api_key = api_key or settings.vllm_api_key
        self.model = model or settings.model_name
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "VLLMClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        guided_json: dict | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Send a chat completion request and return (response_text, meta).

        meta keys:
            prompt_tokens, completion_tokens, total_tokens,
            finish_reason, latency_s, tokens_per_sec

        guided_json: JSON schema dict passed to vLLM's guided_json parameter.
            When set, vLLM uses outlines to constrain decoding to valid JSON
            matching the schema — parse failures become impossible.
            Requires vLLM >= 0.4.0.
        """
        if self._client is None:
            raise RuntimeError("Use VLLMClient as an async context manager.")

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or settings.max_new_tokens,
            "temperature": temperature if temperature is not None else settings.temperature,
            "stream": False,
        }
        if guided_json is not None:
            payload["guided_json"] = guided_json

        t0 = time.perf_counter()
        try:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                "vLLM request failed",
                extra={"status": e.response.status_code, "body": e.response.text[:200]},
            )
            raise
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to vLLM at {self.base_url}. "
                "Is vLLM running? Check: docker compose up vllm"
            )
        elapsed = time.perf_counter() - t0

        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        meta = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens", 0),
            "finish_reason": choice.get("finish_reason"),
            "latency_s": round(elapsed, 3),
            "tokens_per_sec": round(completion_tokens / elapsed, 1) if elapsed > 0 else 0.0,
        }

        if choice.get("finish_reason") not in _STOP_REASONS and choice.get("finish_reason") is not None:
            logger.warning(
                "unexpected finish reason",
                extra={"finish_reason": choice["finish_reason"], **meta},
            )

        logger.info("vLLM chat completed", extra=meta)
        return content, meta

    async def health(self) -> bool:
        """Return True if the vLLM server is reachable and healthy."""
        if self._client is None:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0) as client:
                try:
                    r = await client.get("/health")
                    return r.status_code == 200
                except (httpx.ConnectError, httpx.TimeoutException):
                    return False
        try:
            r = await self._client.get("/health")
            return r.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def list_models(self) -> list[str]:
        """Return model IDs registered with the vLLM server."""
        if self._client is None:
            raise RuntimeError("Use VLLMClient as an async context manager.")
        resp = await self._client.get("/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
