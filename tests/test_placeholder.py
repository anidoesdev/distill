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


def test_example_to_messages_structure() -> None:
    from extractor.data.tokenize import example_to_messages
    from extractor.prompt import EXTRACTION_SYSTEM_PROMPT

    example = {
        "section_text": "We trained a model on ImageNet.",
        "extraction": {
            "authors": ["Jane Smith"],
            "methodology": "CNN trained with SGD.",
            "datasets_used": ["ImageNet"],
            "key_findings": ["Accuracy 73%."],
            "limitations": [],
            "statistical_tests": [],
        },
    }
    messages = example_to_messages(example)

    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == EXTRACTION_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert "ImageNet" in messages[1]["content"]
    assert messages[2]["role"] == "assistant"
    # Assistant response must be valid JSON
    import json
    parsed = json.loads(messages[2]["content"])
    assert parsed["authors"] == ["Jane Smith"]


def test_format_assistant_response_is_compact_json() -> None:
    from extractor.data.tokenize import format_assistant_response

    extraction = {
        "authors": ["A", "B"],
        "methodology": "SVM",
        "datasets_used": [],
        "key_findings": ["F1"],
        "limitations": [],
        "statistical_tests": [],
    }
    result = format_assistant_response(extraction)
    assert "\n" not in result  # compact, no newlines
    import json
    parsed = json.loads(result)
    assert parsed["authors"] == ["A", "B"]
