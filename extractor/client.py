"""Python client SDK for the EXTRACTOR API.

Intended for use by downstream systems such as scientific-rag-assistant.
Provides both async and sync interfaces; the sync variant wraps the async
one via asyncio.run() so it works in scripts and notebooks without an
event loop.

Async usage:
    async with ExtractorClient("http://localhost:8080", api_key="...") as c:
        result = await c.extract(section_text)
        results = await c.extract_batch(["section 1 ...", "section 2 ..."])

Sync usage:
    client = ExtractorClient("http://localhost:8080")
    result = client.extract_sync(section_text)

The client is independent of extractor.config — it only needs a base_url.
All response types are plain dataclasses so no Pydantic dependency is required
in the consuming project.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class ExtractionResponse:
    """Structured response from one /api/extract call."""

    authors: list[str] = field(default_factory=list)
    methodology: str = ""
    datasets_used: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    statistical_tests: list[str] = field(default_factory=list)

    parse_error: str | None = None
    repair_attempted: bool = False
    repair_attempts: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return (
            not self.authors
            and not self.methodology
            and not self.datasets_used
            and not self.key_findings
            and not self.limitations
            and not self.statistical_tests
        )

    @classmethod
    def from_api_response(cls, body: dict[str, Any]) -> "ExtractionResponse":
        extraction = body.get("extraction", {})
        return cls(
            authors=extraction.get("authors", []),
            methodology=extraction.get("methodology", ""),
            datasets_used=extraction.get("datasets_used", []),
            key_findings=extraction.get("key_findings", []),
            limitations=extraction.get("limitations", []),
            statistical_tests=extraction.get("statistical_tests", []),
            parse_error=body.get("parse_error"),
            repair_attempted=body.get("repair_attempted", False),
            repair_attempts=body.get("repair_attempts", 0),
            latency_s=body.get("latency_s", 0.0),
            prompt_tokens=body.get("prompt_tokens", 0),
            completion_tokens=body.get("completion_tokens", 0),
        )


@dataclass
class BatchExtractionResponse:
    """Response from one /api/extract/batch call."""

    results: list[ExtractionResponse] = field(default_factory=list)
    n: int = 0
    failed: int = 0
    latency_s: float = 0.0

    @property
    def success_rate(self) -> float:
        return (self.n - self.failed) / self.n if self.n > 0 else 0.0


class ExtractorClient:
    """Async client for the EXTRACTOR HTTP API.

    Handles auth, retries on transient 5xx errors, and typed responses.

    Args:
        base_url: Base URL of the extractor API (e.g., "http://localhost:8080").
        api_key: Bearer token. Leave empty if auth is disabled on the server.
        timeout: Per-request timeout in seconds.
        max_retries: Number of retries on 5xx responses (exponential backoff).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str = "",
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ExtractorClient":
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _post_with_retry(self, path: str, payload: dict) -> dict:
        if self._client is None:
            raise RuntimeError("Use ExtractorClient as an async context manager.")
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(path, json=payload)
                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp.json()
                # 5xx — wait and retry
                wait = 2 ** attempt
                await asyncio.sleep(wait)
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
            except httpx.TimeoutException as exc:
                last_exc = exc
                await asyncio.sleep(2 ** attempt)
        raise last_exc  # type: ignore[misc]

    async def extract(
        self,
        section_text: str,
        max_tokens: int = 512,
    ) -> ExtractionResponse:
        """Extract structured information from a single paper section."""
        body = await self._post_with_retry(
            "/api/extract",
            {"section_text": section_text, "max_tokens": max_tokens},
        )
        return ExtractionResponse.from_api_response(body)

    async def extract_batch(
        self,
        sections: list[str],
        max_tokens: int = 512,
    ) -> BatchExtractionResponse:
        """Extract structured information from multiple paper sections.

        Uses the /api/extract/batch endpoint (up to 20 sections per call).
        For larger lists, the method automatically splits into chunks.
        """
        CHUNK = 20
        all_results: list[ExtractionResponse] = []
        total_failed = 0
        total_latency = 0.0

        for i in range(0, len(sections), CHUNK):
            chunk = sections[i : i + CHUNK]
            body = await self._post_with_retry(
                "/api/extract/batch",
                {"sections": chunk, "max_tokens": max_tokens},
            )
            for item in body.get("results", []):
                all_results.append(ExtractionResponse.from_api_response(item))
            total_failed += body.get("failed", 0)
            total_latency += body.get("latency_s", 0.0)

        return BatchExtractionResponse(
            results=all_results,
            n=len(all_results),
            failed=total_failed,
            latency_s=round(total_latency, 3),
        )

    async def health(self) -> bool:
        """Return True if the extractor API is reachable."""
        try:
            if self._client:
                r = await self._client.get("/health")
            else:
                async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0) as c:
                    r = await c.get("/health")
            return r.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # ── Sync convenience wrappers ─────────────────────────────────────────────

    def extract_sync(
        self,
        section_text: str,
        max_tokens: int = 512,
    ) -> ExtractionResponse:
        """Synchronous wrapper around extract(). Suitable for scripts/notebooks."""
        return asyncio.run(self._extract_one(section_text, max_tokens))

    def extract_batch_sync(
        self,
        sections: list[str],
        max_tokens: int = 512,
    ) -> BatchExtractionResponse:
        """Synchronous wrapper around extract_batch()."""
        return asyncio.run(self._extract_batch_one(sections, max_tokens))

    async def _extract_one(self, section_text: str, max_tokens: int) -> ExtractionResponse:
        async with self:
            return await self.extract(section_text, max_tokens)

    async def _extract_batch_one(self, sections: list[str], max_tokens: int) -> BatchExtractionResponse:
        async with self:
            return await self.extract_batch(sections, max_tokens)
