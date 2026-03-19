"""Evaluation metrics for structured extraction.

Session 3:  schema_validity_rate, field_presence_rates  (zero-shot baseline)
Session 12: per_field_exact_match, list_field_f1        (post-SFT evaluation)

Keeping them in one module makes it easy to run the same eval suite against
the base model, the SFT model, and GPT-4o-mini in session 13.
"""

from __future__ import annotations

import re
import string

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

LIST_FIELDS = {"authors", "datasets_used", "key_findings", "limitations", "statistical_tests"}
STR_FIELDS = {"methodology"}


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


# ── Session 12 metrics ────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _normalize_set(items: list[str]) -> set[str]:
    return {_normalize(s) for s in items if s.strip()}


def per_field_exact_match(
    predictions: list[ExtractionResult],
    references: list[ExtractionResult],
) -> dict[str, float]:
    """Exact-match accuracy per field.

    List fields: sorted normalized set equality.
    Str fields: normalized string equality.
    Returns per-field fraction in [0, 1].
    """
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(references)} references")

    correct: dict[str, int] = {f: 0 for f in FIELDS}
    n = len(predictions)

    for pred, ref in zip(predictions, references):
        for field in LIST_FIELDS:
            pred_set = _normalize_set(getattr(pred, field))
            ref_set = _normalize_set(getattr(ref, field))
            if pred_set == ref_set:
                correct[field] += 1
        for field in STR_FIELDS:
            if _normalize(getattr(pred, field)) == _normalize(getattr(ref, field)):
                correct[field] += 1

    return {f: round(correct[f] / n, 4) if n else 0.0 for f in FIELDS}


def _prf1_one(pred_items: list[str], ref_items: list[str]) -> tuple[float, float, float]:
    """Precision, recall, F1 for a single example's list field."""
    pred_set = _normalize_set(pred_items)
    ref_set = _normalize_set(ref_items)

    if not pred_set and not ref_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0
    if not ref_set:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & ref_set)
    precision = tp / len(pred_set)
    recall = tp / len(ref_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def list_field_f1(
    predictions: list[ExtractionResult],
    references: list[ExtractionResult],
    field: str,
) -> dict[str, float]:
    """Macro-averaged precision, recall, F1 for a list field.

    Each item is treated as a set element. A predicted item matches a reference
    item if their normalized forms are equal. Macro-averaging means every
    example contributes equally regardless of list length.

    Returns {"precision": float, "recall": float, "f1": float}.
    """
    if field not in LIST_FIELDS:
        raise ValueError(f"'{field}' is not a list field. List fields: {sorted(LIST_FIELDS)}")
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: {len(predictions)} vs {len(references)}")

    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    totals = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for pred, ref in zip(predictions, references):
        p, r, f = _prf1_one(getattr(pred, field), getattr(ref, field))
        totals["precision"] += p
        totals["recall"] += r
        totals["f1"] += f

    n = len(predictions)
    return {k: round(v / n, 4) for k, v in totals.items()}


def eval_suite(
    predictions: list[ExtractionResult],
    references: list[ExtractionResult],
    parse_errors: list[str | None],
) -> dict:
    """Run the full eval suite and return a single results dict.

    Combines validity, exact match, and per-field F1.
    """
    return {
        "n": len(predictions),
        "schema_validity_rate": round(schema_validity_rate(parse_errors), 4),
        "per_field_exact_match": per_field_exact_match(predictions, references),
        "list_field_f1": {
            f: list_field_f1(predictions, references, f) for f in LIST_FIELDS
        },
    }
