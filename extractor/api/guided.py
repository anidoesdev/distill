"""Outlines-based local guided decoding for non-vLLM inference.

When vLLM is not available (local dev, CPU-only machines), this module
provides schema-constrained generation using the `outlines` library.
Outlines compiles a DFA from a Pydantic JSON schema and uses it to mask
invalid tokens at each decode step — the same technique vLLM uses internally
when you pass `guided_json`, but running inside the same process.

Usage:
    from extractor.api.guided import guided_extract
    result, meta = await guided_extract(section_text)

Requires:
    pip install outlines transformers torch

The model is loaded once at module level on first call and cached for the
lifetime of the process. Set MODEL_NAME env var (or EXTRACTOR_MODEL_NAME in
.env) to override the default.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from extractor.config import settings
from extractor.prompt import build_messages
from extractor.schemas.extraction import ExtractionResult
from extractor.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level cache — loaded once, reused across requests
_model = None
_generator = None


def _get_generator():
    """Lazily load the outlines generator (model + schema constraint)."""
    global _model, _generator
    if _generator is not None:
        return _generator

    try:
        import outlines
        import outlines.models as om
    except ImportError as exc:
        raise ImportError(
            "outlines is required for local guided decoding. "
            "Install it: pip install outlines"
        ) from exc

    logger.info(
        "loading model for guided decoding",
        extra={"model": settings.model_name},
    )
    t0 = time.perf_counter()
    _model = om.transformers(
        settings.model_name,
        device="auto",
        model_kwargs={"torch_dtype": "auto"},
    )
    schema = ExtractionResult.model_json_schema()
    _generator = outlines.generate.json(_model, schema)
    elapsed = time.perf_counter() - t0
    logger.info("guided generator ready", extra={"load_s": round(elapsed, 2)})
    return _generator


def _build_prompt_string(messages: list[dict[str, str]]) -> str:
    """Flatten chat messages into a single string for outlines.

    outlines works directly with HuggingFace models and expects a raw string
    prompt, not a chat messages list. We apply the tokenizer's chat template
    if available, otherwise fall back to a simple concatenation.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(settings.model_name)
        if hasattr(tok, "apply_chat_template"):
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception:
        pass

    # Fallback: naive concatenation
    parts = []
    for msg in messages:
        role = msg["role"].upper()
        parts.append(f"[{role}] {msg['content']}")
    parts.append("[ASSISTANT]")
    return "\n".join(parts)


async def guided_extract(
    section_text: str,
    max_tokens: int | None = None,
) -> tuple[ExtractionResult, dict[str, Any]]:
    """Run schema-constrained local inference and return (result, meta).

    This is the non-vLLM path — it runs the model in-process via outlines.
    Runs in a thread executor so it does not block the FastAPI event loop.

    meta keys: prompt_tokens (0 — not tracked), completion_tokens (0),
               latency_s, guided (True)
    """
    messages = build_messages(section_text)
    prompt = _build_prompt_string(messages)

    def _run() -> tuple[ExtractionResult, float]:
        gen = _get_generator()
        t0 = time.perf_counter()
        result_dict = gen(
            prompt,
            max_tokens=max_tokens or settings.max_new_tokens,
            temperature=settings.temperature if settings.temperature > 0 else None,
            sampler=None if settings.temperature == 0 else "multinomial",
        )
        elapsed = time.perf_counter() - t0
        return result_dict, elapsed

    loop = asyncio.get_event_loop()
    result_dict, elapsed = await loop.run_in_executor(None, _run)

    # outlines returns a dict matching the schema directly — no JSON parsing needed
    extraction = ExtractionResult(**result_dict)

    meta: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_s": round(elapsed, 3),
        "guided": True,
        "repair_attempted": False,
        "repair_attempts": 0,
    }
    logger.info("guided extraction complete", extra={"latency_s": meta["latency_s"]})
    return extraction, meta
