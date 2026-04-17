"""Retry+repair logic for malformed model outputs.

When the model produces output that fails JSON parsing, we have two options:
  1. Return an error immediately (bad UX — the caller gets nothing)
  2. Attempt to repair the output and retry with a correction prompt

This module implements option 2. The repair strategy:
  - Round 1: send the raw output back to the model with a "fix this JSON" prompt
  - Round 2: if still broken, try extracting the longest valid JSON substring

The repair prompt is deliberately minimal — we don't tell the model what was wrong
in detail, because the malformed output may contain hallucinated schema fields that
a detailed prompt would preserve. We just ask for valid JSON matching the schema.
"""

from __future__ import annotations

from extractor.schemas.extraction import ExtractionResult
from extractor.utils.logging import get_logger

logger = get_logger(__name__)

REPAIR_SYSTEM_PROMPT = (
    "You are a JSON repair assistant. "
    "The user will provide malformed text that should be valid JSON matching "
    'this schema: {"authors": [str], "methodology": str, "datasets_used": [str], '
    '"key_findings": [str], "limitations": [str], "statistical_tests": [str]}. '
    "Return ONLY valid JSON with exactly these six keys. No explanation, no markdown."
)


def build_repair_messages(broken_output: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "This output failed JSON parsing. Fix it and return valid JSON only:\n\n"
                f"{broken_output[:2000]}"
            ),
        },
    ]


async def repair_and_parse(
    broken_output: str,
    client,  # VLLMClient instance, already entered
) -> tuple[ExtractionResult, str | None]:
    """Attempt one repair call and parse the result.

    Returns (result, error). If repair also fails, returns empty result + error.
    """
    try:
        repair_messages = build_repair_messages(broken_output)
        repaired_raw, meta = await client.chat(repair_messages, max_tokens=512)
        result, error = ExtractionResult.from_model_output(repaired_raw)
        if error:
            logger.warning("repair attempt also failed", extra={"error": error})
        else:
            logger.info("repair succeeded", extra={"completion_tokens": meta.get("completion_tokens")})
        return result, error
    except Exception as exc:
        logger.error("repair call failed", extra={"error": str(exc)})
        return ExtractionResult(), str(exc)


async def extract_with_retry(
    messages: list[dict[str, str]],
    client,
    max_retries: int = 2,
) -> tuple[ExtractionResult, str | None, dict]:
    """Run extraction with up to max_retries repair attempts on parse failure.

    Returns (result, final_error, meta).
      - result is always an ExtractionResult (may be empty on total failure)
      - final_error is None on success, a string on failure
      - meta is from the last model call
    """
    raw, meta = await client.chat(messages)
    result, error = ExtractionResult.from_model_output(raw)

    if error is None:
        return result, None, meta

    logger.warning(
        "parse failed, attempting repair",
        extra={"error": error, "raw_preview": raw[:100]},
    )
    meta["parse_error"] = error
    meta["repair_attempted"] = True

    for attempt in range(max_retries):
        result, error = await repair_and_parse(raw, client)
        meta["repair_attempts"] = attempt + 1
        if error is None:
            return result, None, meta
        # For the next repair attempt, use the latest (still broken) output
        # rather than the original — the repair may have partially improved it

    logger.error(
        "all repair attempts failed",
        extra={"attempts": max_retries, "final_error": error},
    )
    return result, error, meta
