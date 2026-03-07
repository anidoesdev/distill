"""Dataset validation logic for the distilled training set.

Each validator is a function: (example: dict) -> list[str]
Return value is a list of failure reasons. Empty list = passes.

Adding a new check is one function. Removing one is deleting a function.
The CLI script composes all validators defined in ALL_VALIDATORS.
"""

from __future__ import annotations

import re
from typing import Callable

from extractor.schemas.extraction import ExtractionResult

ValidatorFn = Callable[[dict], list[str]]

# Thresholds — adjust based on distribution analysis after first full run
MIN_METHODOLOGY_WORDS = 10
MIN_KEY_FINDINGS = 1
MAX_FIELD_STRING_LEN = 2000
MAX_LIST_FIELD_ITEMS = 30

# Phrases from the system prompt. If any appear in field values,
# the model echoed the prompt instead of extracting.
_INJECTION_MARKERS = [
    "extract structured information",
    "return only valid json",
    "list of author full names",
    "one to three sentence description",
    "list of dataset or benchmark names",
    "list of main results",
    "list of limitations",
    "list of statistical tests",
    "do not infer information",
]


# ── Layer 1: Schema ───────────────────────────────────────────────────────────

def validate_schema(example: dict) -> list[str]:
    """The extraction field must be a valid ExtractionResult dict."""
    try:
        ExtractionResult.model_validate(example.get("extraction", {}))
        return []
    except Exception as e:
        return [f"schema: {e}"]


# ── Layer 2: Semantic ─────────────────────────────────────────────────────────

def validate_non_empty(example: dict) -> list[str]:
    """At least one field must have a non-empty value."""
    ext = example.get("extraction", {})
    result = ExtractionResult.model_validate(ext)
    if result.is_empty():
        return ["semantic: all fields empty"]
    return []


def validate_author_quality(example: dict) -> list[str]:
    """Authors should look like person names, not template text or garbage."""
    reasons = []
    authors: list[str] = example.get("extraction", {}).get("authors", [])

    for author in authors:
        # Must have at least two characters (no single-letter entries)
        if len(author) < 2:
            reasons.append(f"semantic: author too short: {author!r}")
            continue
        # Should not contain digit sequences (no arxiv IDs or DOIs as author names)
        if re.search(r"\d{4,}", author):
            reasons.append(f"semantic: author contains digit sequence: {author!r}")
        # Should not be all uppercase (likely a section header leaked in)
        if author == author.upper() and len(author) > 3:
            reasons.append(f"semantic: author all-caps: {author!r}")

    return reasons


def validate_methodology_length(example: dict) -> list[str]:
    """Methodology should have enough words to be meaningful."""
    methodology: str = example.get("extraction", {}).get("methodology", "")
    if not methodology:
        return []  # Empty is allowed; the schema does not require it
    word_count = len(methodology.split())
    if word_count < MIN_METHODOLOGY_WORDS:
        return [
            f"semantic: methodology too short ({word_count} words, "
            f"min {MIN_METHODOLOGY_WORDS})"
        ]
    return []


def validate_findings_present(example: dict) -> list[str]:
    """A paper abstract should yield at least one key finding."""
    findings: list[str] = example.get("extraction", {}).get("key_findings", [])
    if len(findings) < MIN_KEY_FINDINGS:
        return [
            f"semantic: key_findings empty (min {MIN_KEY_FINDINGS} required)"
        ]
    return []


def validate_field_lengths(example: dict) -> list[str]:
    """No single field value should be absurdly long (indicates runaway generation)."""
    reasons = []
    ext = example.get("extraction", {})

    if len(ext.get("methodology", "")) > MAX_FIELD_STRING_LEN:
        reasons.append("semantic: methodology exceeds max length")

    for list_field in ("authors", "datasets_used", "key_findings", "limitations", "statistical_tests"):
        lst: list = ext.get(list_field, [])
        if len(lst) > MAX_LIST_FIELD_ITEMS:
            reasons.append(f"semantic: {list_field} has {len(lst)} items (max {MAX_LIST_FIELD_ITEMS})")
        for item in lst:
            if len(str(item)) > MAX_FIELD_STRING_LEN:
                reasons.append(f"semantic: item in {list_field} exceeds max length")
                break

    return reasons


# ── Layer 3: Content injection ────────────────────────────────────────────────

def validate_no_content_injection(example: dict) -> list[str]:
    """Field values must not contain phrases from the system prompt.

    This catches the failure mode where the model echoes the instruction
    instead of extracting from the paper.
    """
    ext = example.get("extraction", {})
    all_text = " ".join(
        [ext.get("methodology", "")]
        + ext.get("authors", [])
        + ext.get("datasets_used", [])
        + ext.get("key_findings", [])
        + ext.get("limitations", [])
        + ext.get("statistical_tests", [])
    ).lower()

    for marker in _INJECTION_MARKERS:
        if marker in all_text:
            return [f"injection: prompt text found in output: {marker!r}"]
    return []


# ── Layer 4: Cross-reference ──────────────────────────────────────────────────

def validate_author_cross_reference(example: dict) -> list[str]:
    """At least one extracted author should share a last name with an arXiv author.

    Only applied when the example has arXiv author metadata. Fuzzy: compares
    lowercase last names (last token of each name string).
    """
    arxiv_authors: list[str] = example.get("authors", [])  # from arXiv metadata
    extracted_authors: list[str] = example.get("extraction", {}).get("authors", [])

    if not arxiv_authors or not extracted_authors:
        return []  # Can't cross-reference without both sides

    arxiv_last_names = {
        name.split()[-1].lower().strip(".,") for name in arxiv_authors if name.split()
    }
    extracted_last_names = {
        name.split()[-1].lower().strip(".,") for name in extracted_authors if name.split()
    }

    overlap = arxiv_last_names & extracted_last_names
    if not overlap:
        return [
            f"crossref: no author last-name overlap "
            f"(arxiv={sorted(arxiv_last_names)[:3]}, "
            f"extracted={sorted(extracted_last_names)[:3]})"
        ]
    return []


# ── Composed validator list ───────────────────────────────────────────────────

ALL_VALIDATORS: list[ValidatorFn] = [
    validate_schema,
    validate_non_empty,
    validate_author_quality,
    validate_methodology_length,
    validate_findings_present,
    validate_field_lengths,
    validate_no_content_injection,
    validate_author_cross_reference,
]


def run_validators(
    example: dict,
    validators: list[ValidatorFn] | None = None,
) -> list[str]:
    """Run all validators against one example. Return all failure reasons."""
    vlist = validators if validators is not None else ALL_VALIDATORS
    reasons: list[str] = []
    for fn in vlist:
        reasons.extend(fn(example))
    return reasons
