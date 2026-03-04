"""Evaluation metrics for structured extraction.

Session 3:  schema_validity_rate, field_presence_rates  (zero-shot baseline)
Session 12: per_field_exact_match, list_field_f1        (post-SFT evaluation)

Keeping them in one module makes it easy to run the same eval suite against
the base model, the SFT model, and GPT-4o-mini in session 13.
"""

from __future__ import annotations

from extractor.schemas.extraction import ExtractionResult

# ── Session 3 metrics ─────────────────────────────────────────────────────────

FIELDS = [
    "authors",
    "methodology",
    "datasets_used",
    "key_findings",
    "limitations",
    "statistical_tests",
]


def schema_validity_rate(parse_errors: list[str | None]) -> float:
    """Fraction of outputs that parsed without error.

    Args:
        parse_errors: List of error strings from ExtractionResult.from_model_output.
                      None means success.
    """
    if not parse_errors:
        return 0.0
    return sum(1 for e in parse_errors if e is None) / len(parse_errors)


def field_presence_rates(results: list[ExtractionResult]) -> dict[str, float]:
    """For each field, fraction of results where the field has a non-empty value.

    A field being non-empty does NOT mean it is correct — it means the model
    at least attempted to fill it. Used only for zero-shot baseline analysis.
    """
    if not results:
        return {f: 0.0 for f in FIELDS}
    counts: dict[str, int] = {f: 0 for f in FIELDS}
    for result in results:
        for field, present in result.field_presence().items():
            if present:
                counts[field] += 1
    n = len(results)
    return {f: round(counts[f] / n, 3) for f in FIELDS}


# ── Session 12 metrics (stubs — filled in after SFT) ─────────────────────────

def per_field_exact_match(
    predictions: list[ExtractionResult],
    references: list[ExtractionResult],
) -> dict[str, float]:
    """Exact-match accuracy per field.

    For str fields: case-insensitive string equality.
    For list fields: fraction of predictions where the sorted list exactly
    matches the sorted reference list.
    """
    # Implemented in session 12.
    raise NotImplementedError("Implemented in session 12.")


def list_field_f1(
    predictions: list[ExtractionResult],
    references: list[ExtractionResult],
    field: str,
) -> dict[str, float]:
    """Macro-averaged precision, recall, F1 for a list field.

    Each item in the list is treated as a set element. A predicted item is
    considered correct if it appears in the reference list (case-insensitive).
    """
    # Implemented in session 12.
    raise NotImplementedError("Implemented in session 12.")
