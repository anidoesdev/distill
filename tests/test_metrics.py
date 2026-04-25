"""Comprehensive tests for extractor.eval.metrics.

Covers edge cases that the session 12 tests in test_placeholder.py
didn't reach: single-item lists, all-wrong predictions, cross-field
independence, alignment_tax, and regression contracts.
"""

import pytest

from extractor.eval.metrics import (
    alignment_tax,
    eval_suite,
    list_field_f1,
    per_field_exact_match,
)
from extractor.schemas.extraction import ExtractionResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def make(
    authors=None,
    methodology="",
    datasets_used=None,
    key_findings=None,
    limitations=None,
    statistical_tests=None,
) -> ExtractionResult:
    return ExtractionResult(
        authors=authors or [],
        methodology=methodology,
        datasets_used=datasets_used or [],
        key_findings=key_findings or [],
        limitations=limitations or [],
        statistical_tests=statistical_tests or [],
    )


# ── per_field_exact_match edge cases ─────────────────────────────────────────

def test_em_all_wrong():
    pred = make(authors=["Alice"], methodology="CNN")
    ref = make(authors=["Bob"], methodology="SVM")
    result = per_field_exact_match([pred], [ref])
    assert result["authors"] == 0.0
    assert result["methodology"] == 0.0


def test_em_multiple_examples_averaging():
    preds = [make(authors=["A"]), make(authors=["B"]), make(authors=["C"])]
    refs  = [make(authors=["A"]), make(authors=["B"]), make(authors=["X"])]
    result = per_field_exact_match(preds, refs)
    # 2 correct out of 3 = 0.6667 (rounded to 4dp by the implementation)
    assert abs(result["authors"] - 2/3) < 1e-3


def test_em_both_empty_list_counts_as_match():
    pred = make(datasets_used=[])
    ref = make(datasets_used=[])
    result = per_field_exact_match([pred], [ref])
    assert result["datasets_used"] == 1.0


def test_em_one_empty_one_not_is_mismatch():
    pred = make(datasets_used=["ImageNet"])
    ref = make(datasets_used=[])
    result = per_field_exact_match([pred], [ref])
    assert result["datasets_used"] == 0.0


def test_em_list_order_independent():
    pred = make(key_findings=["F2", "F1"])
    ref = make(key_findings=["F1", "F2"])
    result = per_field_exact_match([pred], [ref])
    assert result["key_findings"] == 1.0


def test_em_fields_are_independent():
    """Getting methodology right should not affect authors score."""
    pred = make(authors=["Wrong"], methodology="CNN")
    ref = make(authors=["Alice"], methodology="CNN")
    result = per_field_exact_match([pred], [ref])
    assert result["methodology"] == 1.0
    assert result["authors"] == 0.0


# ── list_field_f1 edge cases ──────────────────────────────────────────────────

def test_f1_empty_pred_non_empty_ref():
    # _prf1_one: empty pred_set → returns (0, 0, 0) — no vacuous precision
    pred = make(authors=[])
    ref = make(authors=["Alice", "Bob"])
    result = list_field_f1([pred], [ref], "authors")
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_f1_non_empty_pred_empty_ref():
    # _prf1_one: empty ref_set → returns (0, 0, 0) — no vacuous recall
    pred = make(authors=["Alice"])
    ref = make(authors=[])
    result = list_field_f1([pred], [ref], "authors")
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_f1_single_item_match():
    pred = make(datasets_used=["ImageNet"])
    ref = make(datasets_used=["ImageNet"])
    result = list_field_f1([pred], [ref], "datasets_used")
    assert result["f1"] == 1.0


def test_f1_case_insensitive():
    pred = make(datasets_used=["imagenet"])
    ref = make(datasets_used=["ImageNet"])
    result = list_field_f1([pred], [ref], "datasets_used")
    assert result["f1"] == 1.0


def test_f1_duplicate_predictions_counted_once():
    """Predicting the same author twice should not inflate precision."""
    pred = make(authors=["Alice", "Alice"])
    ref = make(authors=["Alice"])
    result = list_field_f1([pred], [ref], "authors")
    assert result["precision"] == 1.0


def test_f1_macro_average_across_examples():
    # Example 1: perfect (F1=1.0), Example 2: zero overlap (F1=0.0)
    preds = [make(authors=["Alice"]), make(authors=["Bob"])]
    refs =  [make(authors=["Alice"]), make(authors=["Charlie"])]
    result = list_field_f1(preds, refs, "authors")
    assert abs(result["f1"] - 0.5) < 1e-9


# ── eval_suite ────────────────────────────────────────────────────────────────

def test_eval_suite_perfect_result(sample_result):
    result = eval_suite([sample_result], [sample_result], [None])
    assert result["schema_validity_rate"] == 1.0
    assert result["macro_em"] == 1.0
    assert result["macro_f1"] == 1.0


def test_eval_suite_schema_validity_rate():
    # Two results: one valid (from model_validate), one empty (parse failure)
    valid = ExtractionResult(methodology="CNN")
    invalid = ExtractionResult()  # empty — is_empty() True
    result = eval_suite([valid, invalid], [valid, valid], [None, None])
    # Both pass schema_validity (ExtractionResult always validates)
    assert result["schema_validity_rate"] == 1.0


def test_eval_suite_n_field():
    preds = [ExtractionResult()] * 5
    refs = [ExtractionResult()] * 5
    result = eval_suite(preds, refs, [None] * 5)
    assert result["n"] == 5


def test_eval_suite_all_keys_present(sample_result):
    result = eval_suite([sample_result], [sample_result], [None])
    expected_keys = {
        "n", "schema_validity_rate", "macro_em", "macro_f1",
        "per_field_exact_match", "list_field_f1",
    }
    assert expected_keys.issubset(set(result.keys()))


def test_eval_suite_per_field_em_is_dict(sample_result):
    result = eval_suite([sample_result], [sample_result], [None])
    pfem = result["per_field_exact_match"]
    assert isinstance(pfem, dict)
    assert "authors" in pfem
    assert "methodology" in pfem


# ── alignment_tax ─────────────────────────────────────────────────────────────

def test_alignment_tax_no_regression():
    # alignment_tax reads macro_f1 from the top-level key, not from list_field_f1
    baseline = {
        "macro_f1": 0.80,
        "macro_em": 0.70,
        "list_field_f1": {"authors": {"f1": 0.8}, "key_findings": {"f1": 0.9}},
        "per_field_exact_match": {},
    }
    dpo = {
        "macro_f1": 0.83,
        "macro_em": 0.73,
        "list_field_f1": {"authors": {"f1": 0.82}, "key_findings": {"f1": 0.91}},
        "per_field_exact_match": {},
    }
    tax = alignment_tax(baseline, dpo)
    assert tax["macro_f1_delta"] > 0
    assert "authors" in tax["improved_fields"]
    assert "key_findings" in tax["improved_fields"]
    assert len(tax["regressed_fields"]) == 0


def test_alignment_tax_regression():
    baseline = {
        "macro_f1": 0.85,
        "macro_em": 0.75,
        "list_field_f1": {"authors": {"f1": 0.9}, "key_findings": {"f1": 0.8}},
        "per_field_exact_match": {},
    }
    dpo = {
        "macro_f1": 0.72,
        "macro_em": 0.65,
        "list_field_f1": {"authors": {"f1": 0.7}, "key_findings": {"f1": 0.75}},
        "per_field_exact_match": {},
    }
    tax = alignment_tax(baseline, dpo)
    assert tax["macro_f1_delta"] < 0
    assert "authors" in tax["regressed_fields"]
    assert "key_findings" in tax["regressed_fields"]


def test_alignment_tax_mixed():
    baseline = {
        "macro_f1": 0.75,
        "macro_em": 0.70,
        "list_field_f1": {"authors": {"f1": 0.8}, "key_findings": {"f1": 0.7}},
        "per_field_exact_match": {},
    }
    dpo = {
        "macro_f1": 0.73,
        "macro_em": 0.69,
        "list_field_f1": {"authors": {"f1": 0.85}, "key_findings": {"f1": 0.6}},
        "per_field_exact_match": {},
    }
    tax = alignment_tax(baseline, dpo)
    assert "authors" in tax["improved_fields"]
    assert "key_findings" in tax["regressed_fields"]


def test_alignment_tax_contains_expected_keys():
    baseline = {"list_field_f1": {"authors": {"f1": 0.8}}}
    dpo      = {"list_field_f1": {"authors": {"f1": 0.8}}}
    tax = alignment_tax(baseline, dpo)
    expected = {
        "per_field_f1_delta", "macro_f1_delta",
        "improved_fields", "regressed_fields", "neutral_fields",
    }
    assert expected.issubset(set(tax.keys()))


# ── Regression: output shape contracts ───────────────────────────────────────

def test_eval_suite_output_types(sample_result):
    """Regression: eval_suite always returns the same shape, regardless of input."""
    result = eval_suite([sample_result], [sample_result], [None])
    assert isinstance(result["n"], int)
    assert isinstance(result["schema_validity_rate"], float)
    assert isinstance(result["macro_em"], float)
    assert isinstance(result["macro_f1"], float)
    assert isinstance(result["per_field_exact_match"], dict)
    assert isinstance(result["list_field_f1"], dict)
    # All per-field EM values are floats in [0, 1]
    for v in result["per_field_exact_match"].values():
        assert 0.0 <= v <= 1.0


def test_list_field_f1_output_types(sample_result):
    result = list_field_f1([sample_result], [sample_result], "authors")
    assert isinstance(result["precision"], float)
    assert isinstance(result["recall"], float)
    assert isinstance(result["f1"], float)
    assert 0.0 <= result["f1"] <= 1.0
