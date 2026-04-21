"""Batch extraction endpoint.

POST /api/extract/batch
  body: {"sections": ["text1", "text2", ...], "max_tokens": 512}
  returns: {"results": [...], "n": N, "failed": K, "latency_s": X}

All sections are dispatched concurrently to vLLM via asyncio.gather.
vLLM's continuous batching scheduler absorbs the burst — this is one of
the key advantages of vLLM over a standard transformers inference loop.

Limits:
  - Max 20 sections per call (prevents OOM on the vLLM side)
  - Each section follows the same max_tokens constraint as /api/extract

This module is mounted as a router onto the main FastAPI app.
"""

from __future__ import annotations

import asyncio
import time
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from extractor.api.auth import verify_api_key
from extractor.api.repair import extract_with_retry
from extractor.config import settings
from extractor.model.vllm_client import VLLMClient
from extractor.prompt import build_messages
from extractor.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

_MAX_BATCH = 20


class BatchExtractRequest(BaseModel):
    sections: list[str] = Field(
        ...,
        min_length=1,
        max_length=_MAX_BATCH,
        description=f"List of paper section texts. Maximum {_MAX_BATCH} per request.",
        examples=[["We trained a CNN on ImageNet...", "Authors: Smith et al. ..."]],
    )
    max_tokens: int = Field(
        default=512,
        ge=64,
        le=1024,
        description="Maximum tokens per extraction.",
    )


class SingleExtractionResult(BaseModel):
    extraction: dict
    parse_error: str | None = None
    repair_attempted: bool = False
    repair_attempts: int = 0
    latency_s: float
    prompt_tokens: int
    completion_tokens: int


class BatchExtractResponse(BaseModel):
    results: list[SingleExtractionResult]
    n: int
    failed: int
    latency_s: float


async def _extract_one(
    section_text: str,
    client: VLLMClient,
    max_tokens: int,
) -> SingleExtractionResult:
    messages = build_messages(section_text)
    result, error, meta = await extract_with_retry(
        messages,
        client,
        max_retries=settings.max_retries,
    )
    return SingleExtractionResult(
        extraction=result.model_dump(),
        parse_error=error,
        repair_attempted=meta.get("repair_attempted", False),
        repair_attempts=meta.get("repair_attempts", 0),
        latency_s=meta.get("latency_s", 0.0),
        prompt_tokens=meta.get("prompt_tokens", 0),
        completion_tokens=meta.get("completion_tokens", 0),
    )


@router.post("/api/extract/batch", dependencies=[Depends(verify_api_key)])
async def extract_batch(request: BatchExtractRequest) -> BatchExtractResponse:
    """Extract structured JSON from multiple paper sections concurrently.

    All sections are dispatched to vLLM in a single asyncio.gather call.
    Results are returned in the same order as the input sections.
    Sections that fail parse repair return parse_error != null.
    """
    t0 = time.perf_counter()

    async with VLLMClient() as client:
        tasks = [
            _extract_one(text, client, request.max_tokens)
            for text in request.sections
        ]
        results: list[SingleExtractionResult] = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - t0
    failed = sum(1 for r in results if r.parse_error is not None)

    logger.info(
        "batch extraction complete",
        extra={"n": len(results), "failed": failed, "latency_s": round(elapsed, 3)},
    )

    return BatchExtractResponse(
        results=list(results),
        n=len(results),
        failed=failed,
        latency_s=round(elapsed, 3),
    )
