"""Unit tests for extractor.schemas.extraction.ExtractionResult."""

import json

import pytest

from extractor.schemas.extraction import ExtractionResult


# ── Validator: coerce_to_list ─────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (None,            []),
    ([],              []),
    ("Alice",         ["Alice"]),
    ("  Alice  ",     ["Alice"]),          # strips whitespace
    (["Alice", "Bob"], ["Alice", "Bob"]),
    (["Alice", ""],   ["Alice"]),          # empty string filtered
    (["Alice", "  "], ["Alice"]),          # whitespace-only filtered
    (123,             []),                 # non-list/non-str/non-None → empty (validator fallback)
])
def test_coerce_to_list(value, expected):
    result = ExtractionResult(authors=value)
    assert result.authors == expected


def test_coerce_to_list_all_list_fields():
    """All five list fields go through the same validator."""
    fields = ["datasets_used", "key_findings", "limitations", "statistical_tests"]
    for field in fields:
        r = ExtractionResult(**{field: "single string"})
        assert getattr(r, field) == ["single string"], f"failed for field: {field}"


# ── Validator: coerce_to_str ──────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (None,           ""),
    ("",             ""),
    ("  CNN  ",      "CNN"),   # strips whitespace
    (42,             "42"),    # int coerced
])
def test_coerce_to_str(value, expected):
    result = ExtractionResult(methodology=value)
    assert result.methodology == expected


# ── is_empty ──────────────────────────────────────────────────────────────────

def test_is_empty_default():
    assert ExtractionResult().is_empty()


@pytest.mark.parametrize("field,value", [
    ("authors",          ["Alice"]),
    ("methodology",      "CNN"),
    ("datasets_used",    ["ImageNet"]),
    ("key_findings",     ["73% accuracy"]),
    ("limitations",      ["English only"]),
    ("statistical_tests", ["t-test"]),
])
def test_is_empty_false_when_any_field_set(field, value):
    r = ExtractionResult(**{field: value})
    assert not r.is_empty()


# ── field_presence ────────────────────────────────────────────────────────────

def test_field_presence_all_empty():
    r = ExtractionResult()
    fp = r.field_presence()
    assert all(v is False for v in fp.values())
    assert set(fp.keys()) == {
        "authors", "methodology", "datasets_used",
        "key_findings", "limitations", "statistical_tests",
    }


def test_field_presence_partial():
    r = ExtractionResult(authors=["Alice"], methodology="CNN")
    fp = r.field_presence()
    assert fp["authors"] is True
    assert fp["methodology"] is True
    assert fp["datasets_used"] is False


# ── from_model_output: success paths ─────────────────────────────────────────

def test_parse_clean_json(valid_json_response):
    result, error = ExtractionResult.from_model_output(valid_json_response)
    assert error is None
    assert result.authors == ["Alice Smith", "Bob Jones"]
    assert result.methodology.startswith("We trained")


def test_parse_markdown_fenced_json(valid_json_response):
    fenced = f"```json\n{valid_json_response}\n```"
    result, error = ExtractionResult.from_model_output(fenced)
    assert error is None
    assert result.authors == ["Alice Smith", "Bob Jones"]


def test_parse_json_embedded_in_prose(valid_json_response):
    prose = f"Here is the structured extraction:\n\n{valid_json_response}\n\nThat's it."
    result, error = ExtractionResult.from_model_output(prose)
    assert error is None


def test_parse_extra_whitespace(valid_json_response):
    result, error = ExtractionResult.from_model_output(f"  \n{valid_json_response}\n  ")
    assert error is None


# ── from_model_output: failure paths ─────────────────────────────────────────

def test_parse_returns_empty_result_on_failure():
    result, error = ExtractionResult.from_model_output("not json")
    assert error is not None
    assert "JSONDecodeError" in error
    assert result.is_empty()


def test_parse_partial_schema_still_validates():
    """Missing fields default to empty — not a validation error."""
    partial = json.dumps({"authors": ["Alice"], "methodology": "CNN"})
    result, error = ExtractionResult.from_model_output(partial)
    assert error is None
    assert result.authors == ["Alice"]
    assert result.datasets_used == []


def test_parse_extra_keys_ignored():
    """Extra JSON keys are silently ignored by pydantic model_validate."""
    extra = json.dumps({
        "authors": ["Alice"],
        "methodology": "CNN",
        "datasets_used": [],
        "key_findings": [],
        "limitations": [],
        "statistical_tests": [],
        "unknown_field": "should be ignored",
    })
    result, error = ExtractionResult.from_model_output(extra)
    assert error is None
    assert not hasattr(result, "unknown_field")


def test_parse_empty_string_input():
    result, error = ExtractionResult.from_model_output("")
    assert error is not None
    assert result.is_empty()


# ── json_schema_str ───────────────────────────────────────────────────────────

def test_json_schema_str_is_valid_json():
    schema_str = ExtractionResult.json_schema_str()
    schema = json.loads(schema_str)
    assert "properties" in schema
    assert "authors" in schema["properties"]


def test_model_json_schema_has_all_fields():
    schema = ExtractionResult.model_json_schema()
    props = schema["properties"]
    expected = {"authors", "methodology", "datasets_used",
                "key_findings", "limitations", "statistical_tests"}
    assert expected.issubset(set(props.keys()))
