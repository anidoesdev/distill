"""Placeholder — real tests added progressively from session 12 onward."""


def test_package_importable() -> None:
    import extractor  # noqa: F401

    assert extractor.__version__ == "0.1.0"


def test_schema_imports() -> None:
    from extractor.schemas.extraction import ExtractionResult

    result = ExtractionResult()
    assert result.is_empty()
    assert result.authors == []
    assert result.methodology == ""


def test_schema_from_model_output_valid() -> None:
    from extractor.schemas.extraction import ExtractionResult

    raw = '{"authors": ["Jane Smith"], "methodology": "We trained a CNN.", "datasets_used": [], "key_findings": ["Accuracy improved."], "limitations": [], "statistical_tests": []}'
    result, error = ExtractionResult.from_model_output(raw)
    assert error is None
    assert result.authors == ["Jane Smith"]
    assert result.methodology == "We trained a CNN."


def test_schema_from_model_output_with_markdown_fence() -> None:
    from extractor.schemas.extraction import ExtractionResult

    raw = '```json\n{"authors": ["A"], "methodology": "SVM", "datasets_used": [], "key_findings": [], "limitations": [], "statistical_tests": []}\n```'
    result, error = ExtractionResult.from_model_output(raw)
    assert error is None
    assert result.authors == ["A"]


def test_prompt_builds_correctly() -> None:
    from extractor.prompt import EXTRACTION_SYSTEM_PROMPT, build_messages

    messages = build_messages("sample section text")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == EXTRACTION_SYSTEM_PROMPT
    assert "sample section text" in messages[1]["content"]


def test_splits_module_importable() -> None:
    from extractor.data.splits import load_split, verify_no_leakage  # noqa: F401
