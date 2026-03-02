"""Output schema for structured extraction — full definition in session 3.

Placeholder so the package structure is valid now.
"""

from pydantic import BaseModel


class ExtractionResult(BaseModel):
    """Structured extraction output. Fields filled in during session 3."""

    authors: list[str] = []
    methodology: str = ""
    datasets_used: list[str] = []
    key_findings: list[str] = []
    limitations: list[str] = []
    statistical_tests: list[str] = []
