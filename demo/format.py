"""Markdown/HTML formatting helpers for ExtractionResult display in the demo UI.

Converts a structured extraction into human-readable markdown so non-technical
users can read it without staring at raw JSON.
"""

from __future__ import annotations

from extractor.schemas.extraction import ExtractionResult


def result_to_markdown(result: ExtractionResult, parse_error: str | None = None) -> str:
    """Render an ExtractionResult as a formatted markdown string.

    Empty fields are omitted so the output stays compact.
    A parse error banner is prepended when repair failed.
    """
    lines: list[str] = []

    if parse_error:
        lines.append(f"> **⚠ Parse error (partial result):** `{parse_error}`\n")

    if result.authors:
        lines.append("### Authors")
        for author in result.authors:
            lines.append(f"- {author}")
        lines.append("")

    if result.methodology:
        lines.append("### Methodology")
        lines.append(result.methodology)
        lines.append("")

    if result.datasets_used:
        lines.append("### Datasets Used")
        for ds in result.datasets_used:
            lines.append(f"- {ds}")
        lines.append("")

    if result.key_findings:
        lines.append("### Key Findings")
        for i, finding in enumerate(result.key_findings, 1):
            lines.append(f"{i}. {finding}")
        lines.append("")

    if result.limitations:
        lines.append("### Limitations")
        for lim in result.limitations:
            lines.append(f"- {lim}")
        lines.append("")

    if result.statistical_tests:
        lines.append("### Statistical Tests")
        for test in result.statistical_tests:
            lines.append(f"- {test}")
        lines.append("")

    if result.is_empty() and not parse_error:
        return "_No structured information could be extracted from this section._"

    return "\n".join(lines).strip()


def result_to_field_table(result: ExtractionResult) -> str:
    """Render a compact field-presence summary table in markdown."""
    presence = result.field_presence()
    rows = ["| Field | Present | Value preview |", "|-------|---------|---------------|"]
    for field, present in presence.items():
        val = getattr(result, field)
        if isinstance(val, list):
            preview = ", ".join(val[:2]) + ("..." if len(val) > 2 else "")
        else:
            preview = (val[:60] + "...") if len(val) > 60 else val
        icon = "✓" if present else "✗"
        rows.append(f"| {field} | {icon} | {preview or '—'} |")
    return "\n".join(rows)
