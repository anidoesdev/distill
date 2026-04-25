"""FastAPI application — EXTRACTOR API.

Endpoints:
  GET  /health               — liveness probe (no auth)
  GET  /api/info             — model info + vLLM status (auth required)
  POST /api/extract          — extract structured JSON from a paper section (auth required)
  POST /api/extract/batch    — batch extraction, up to 20 sections (auth required)
  GET  /demo                 — Gradio interactive UI (no auth)

Run locally (no model — health endpoint only):
    uvicorn extractor.api.main:app --host 0.0.0.0 --port 8080 --reload

Run with vLLM backend:
    docker compose up
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from extractor.api.auth import verify_api_key
from extractor.api.batch import router as batch_router
from extractor.api.guided import guided_extract
from extractor.api.repair import extract_with_retry
from extractor.config import settings
from extractor.model.vllm_client import VLLMClient
from extractor.prompt import build_messages
from extractor.schemas.extraction import ExtractionResult
from extractor.utils.logging import configure_logging, get_logger

configure_logging(settings.log_level)
logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("extractor api starting", extra={"model": settings.model_name})
    yield
    logger.info("extractor api shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EXTRACTOR",
    description="Fine-tuned scientific paper information extractor",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(batch_router)

# Gradio demo mounted at /demo — imported lazily so the API starts even if
# gradio is not installed (it's an optional dependency for the demo only).
try:
    import gradio as gr
    from demo.app import demo as gradio_demo

    gr.mount_gradio_app(app, gradio_demo, path="/demo")
    logger.info("gradio demo mounted at /demo")
except Exception as _gradio_exc:
    logger.info("gradio demo not available — /demo endpoint disabled (%s)", _gradio_exc)


# ── Request/response logging middleware ───────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0
    logger.info(
        "http request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_s": round(elapsed, 3),
        },
    )
    return response


# ── Schemas ───────────────────────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    section_text: str = Field(
        ...,
        min_length=10,
        max_length=8000,
        description="The paper section text to extract structured information from.",
        examples=["We trained a CNN on ImageNet using SGD with momentum 0.9..."],
    )
    max_tokens: int = Field(
        default=512,
        ge=64,
        le=1024,
        description="Maximum tokens to generate for the extraction.",
    )


class ExtractResponse(BaseModel):
    extraction: dict
    parse_error: str | None = None
    repair_attempted: bool = False
    repair_attempts: int = 0
    latency_s: float
    prompt_tokens: int
    completion_tokens: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/info", dependencies=[Depends(verify_api_key)])
async def info() -> dict:
    """Return model configuration and vLLM health status."""
    async with VLLMClient() as client:
        vllm_healthy = await client.health()
        models = []
        if vllm_healthy:
            try:
                models = await client.list_models()
            except Exception:
                pass

    return {
        "model": settings.model_name,
        "vllm_base_url": settings.vllm_base_url,
        "vllm_healthy": vllm_healthy,
        "models": models,
        "max_new_tokens": settings.max_new_tokens,
        "max_retries": settings.max_retries,
        "auth_enabled": bool(settings.api_key),
    }


@app.post("/api/extract", dependencies=[Depends(verify_api_key)])
async def extract(request: ExtractRequest) -> ExtractResponse:
    """Extract structured JSON from a scientific paper section.

    Returns a JSON object with six fields:
      authors, methodology, datasets_used, key_findings, limitations, statistical_tests

    Two execution paths depending on settings.use_guided_decoding:
      - False (default): calls vLLM, retries on parse failure up to max_retries times
      - True: calls vLLM with guided_json schema constraint (no parse failures possible)
        or falls back to in-process outlines decoding if vLLM is unavailable
    """
    if settings.use_guided_decoding:
        schema = ExtractionResult.model_json_schema()
        messages = build_messages(request.section_text)
        async with VLLMClient() as client:
            vllm_ok = await client.health()
            if vllm_ok:
                raw, meta = await client.chat(
                    messages,
                    max_tokens=request.max_tokens,
                    guided_json=schema,
                )
                result, error = ExtractionResult.from_model_output(raw)
                meta.update({"repair_attempted": False, "repair_attempts": 0})
            else:
                logger.warning("vLLM unreachable, falling back to local outlines decoding")
                result, meta = await guided_extract(
                    request.section_text,
                    max_tokens=request.max_tokens,
                )
                error = None
    else:
        messages = build_messages(request.section_text)
        async with VLLMClient() as client:
            result, error, meta = await extract_with_retry(
                messages,
                client,
                max_retries=settings.max_retries,
            )

    return ExtractResponse(
        extraction=result.model_dump(),
        parse_error=error,
        repair_attempted=meta.get("repair_attempted", False),
        repair_attempts=meta.get("repair_attempts", 0),
        latency_s=meta.get("latency_s", 0.0),
        prompt_tokens=meta.get("prompt_tokens", 0),
        completion_tokens=meta.get("completion_tokens", 0),
    )
