"""Output schema for structured extraction.

This schema is the contract between the model and the application.
Field names and types must not change after training data generation (session 4).
Any change requires regenerating the dataset and retraining.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExtractionResult(BaseModel):
    """Structured extraction output for one scientific paper section.

    All fields default to empty so a failed parse yields a valid (but empty)
    result. Callers check `is_empty()` or `from_model_output` error to detect
    parse failures.
    """

    authors: list[str] = Field(
        default_factory=list,
        description="Full names of paper authors as stated in the paper.",
    )
    methodology: str = Field(
        default="",
        description=(
            "Description of methods, algorithms, or experimental approaches used. "
            "One to three sentences."
        ),
    )
    datasets_used: list[str] = Field(
        default_factory=list,
        description="Names of datasets or benchmarks used for training or evaluation.",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Main results and conclusions. Each finding is one sentence.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations or constraints explicitly acknowledged by the authors.",
    )
    statistical_tests: list[str] = Field(
        default_factory=list,
        description="Statistical tests, significance measures, or evaluation protocols mentioned.",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator(
        "authors",
        "datasets_used",
        "key_findings",
        "limitations",
        "statistical_tests",
        mode="before",
    )
    @classmethod
    def coerce_to_list(cls, v: Any) -> list[str]:
        """Accept None, a bare string, or a list. Filter empty entries."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return []

    @field_validator("methodology", mode="before")
    @classmethod
    def coerce_to_str(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    # ── Parsing ───────────────────────────────────────────────────────────────

    @classmethod
    def from_model_output(cls, text: str) -> tuple[ExtractionResult, str | None]:
        """Parse model output text into an ExtractionResult.

        Handles three common model output formats:
          1. Clean JSON: {"authors": [...], ...}
          2. Markdown-fenced: ```json\n{...}\n```
          3. JSON embedded in prose: "Here is the result: {...}"

        Returns:
            (result, error) — error is None on success. On failure, result is a
            valid but empty ExtractionResult so callers always get a usable object.
        """
        text = text.strip()

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
            if match:
                text = match.group(1).strip()

        # If prose wraps the JSON, extract the outermost {...} object
        brace_open = text.find("{")
        brace_close = text.rfind("}")
        if brace_open > 0 and brace_close > brace_open:
            text = text[brace_open : brace_close + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return cls(), f"JSONDecodeError: {e}"

        try:
            result = cls.model_validate(data)
            return result, None
        except Exception as e:
            return cls(), f"ValidationError: {e}"

    # ── Introspection helpers ─────────────────────────────────────────────────

    def is_empty(self) -> bool:
        """True when all fields are at their defaults — total parse failure."""
        return (
            not self.authors
            and not self.methodology
            and not self.datasets_used
            and not self.key_findings
            and not self.limitations
            and not self.statistical_tests
        )

    def field_presence(self) -> dict[str, bool]:
        """Return which fields have non-empty values."""
        return {
            "authors": bool(self.authors),
            "methodology": bool(self.methodology),
            "datasets_used": bool(self.datasets_used),
            "key_findings": bool(self.key_findings),
            "limitations": bool(self.limitations),
            "statistical_tests": bool(self.statistical_tests),
        }

    @classmethod
    def json_schema_str(cls) -> str:
        """Compact JSON schema string for embedding in prompts."""
        return json.dumps(cls.model_json_schema(), indent=2)
