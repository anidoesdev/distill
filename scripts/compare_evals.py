"""Side-by-side comparison of base, SFT, DPO, and teacher (GPT-4o-mini) models.

Reads the JSON result files written by eval_base_model.py, eval_sft.py,
eval_teacher.py, and optionally eval_sft.py --model-dir checkpoints/dpo-merged.

Usage:
    python scripts/compare_evals.py
    python scripts/compare_evals.py --dpo data/eval/dpo_eval_results.json
    python scripts/compare_evals.py \
        --base    data/eval/base_eval_results.json \
        --sft     data/eval/sft_eval_results.json \
        --dpo     data/eval/dpo_eval_results.json \
        --teacher data/eval/teacher_eval_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("data/eval")

DEFAULT_PATHS = {
    "base":    RESULTS_DIR / "base_eval_results.json",
    "sft":     RESULTS_DIR / "sft_eval_results.json",
    "dpo":     RESULTS_DIR / "dpo_eval_results.json",
    "teacher": RESULTS_DIR / "teacher_eval_results.json",
}

FIELDS = [
    "authors",
    "methodology",
    "datasets_used",
    "key_findings",
    "limitations",
    "statistical_tests",
]

LIST_FIELDS = {"authors", "datasets_used", "key_findings", "limitations", "statistical_tests"}


def load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def macro_f1(metrics: dict) -> float:
    """Average F1 across all list fields."""
    f1s = [metrics["list_field_f1"][f]["f1"] for f in LIST_FIELDS]
    return sum(f1s) / len(f1s)


def macro_em(metrics: dict) -> float:
    """Average exact match across all fields."""
    ems = list(metrics["per_field_exact_match"].values())
    return sum(ems) / len(ems)


def print_comparison(results: dict[str, dict]) -> None:
    labels = list(results.keys())
    if not labels:
        print("No results to compare. Run the eval scripts first.")
        return

    col_w = 10

    print("\n" + "=" * (28 + col_w * len(labels) + 2 * len(labels)))
    print("MODEL COMPARISON")
    print("=" * (28 + col_w * len(labels) + 2 * len(labels)))

    # Header row
    header = f"  {'':26}"
    for label in labels:
        header += f"  {label:>{col_w}}"
    print(header)

    # Model names
    model_row = f"  {'model':26}"
    for label in labels:
        name = results[label].get("model", "?")
        short = name.split("/")[-1][:col_w]
        model_row += f"  {short:>{col_w}}"
    print(model_row)

    # N examples
    n_row = f"  {'n_examples':26}"
    for label in labels:
        n = results[label]["metrics"]["n"]
        n_row += f"  {n:>{col_w}}"
    print(n_row)

    print()

    # Schema validity
    val_row = f"  {'schema_validity':26}"
    for label in labels:
        v = results[label]["metrics"]["schema_validity_rate"]
        val_row += f"  {v:>{col_w}.1%}"
    print(val_row)

    print()

    # Per-field exact match
    print(f"  {'--- EXACT MATCH ---':26}")
    for field in FIELDS:
        row = f"  {field:<26}"
        for label in labels:
            score = results[label]["metrics"]["per_field_exact_match"].get(field, 0.0)
            row += f"  {score:>{col_w}.1%}"
        print(row)

    macro_em_row = f"  {'  macro avg':26}"
    for label in labels:
        score = macro_em(results[label]["metrics"])
        macro_em_row += f"  {score:>{col_w}.1%}"
    print(macro_em_row)

    print()

    # Per-field F1 (list fields only)
    print(f"  {'--- LIST FIELD F1 ---':26}")
    for field in LIST_FIELDS:
        row = f"  {field:<26}"
        for label in labels:
            f1_data = results[label]["metrics"]["list_field_f1"].get(field, {})
            f1 = f1_data.get("f1", 0.0)
            row += f"  {f1:>{col_w}.1%}"
        print(row)

    macro_f1_row = f"  {'  macro avg F1':26}"
    for label in labels:
        score = macro_f1(results[label]["metrics"])
        macro_f1_row += f"  {score:>{col_w}.1%}"
    print(macro_f1_row)

    print("=" * (28 + col_w * len(labels) + 2 * len(labels)))

    # Delta table (SFT vs base, DPO vs SFT, gap to teacher)
    if "base" in results and "sft" in results:
        sft_em = macro_em(results["sft"]["metrics"])
        base_em = macro_em(results["base"]["metrics"])
        delta_em = sft_em - base_em
        sft_f1 = macro_f1(results["sft"]["metrics"])
        base_f1 = macro_f1(results["base"]["metrics"])
        delta_f1 = sft_f1 - base_f1
        sign_em = "+" if delta_em >= 0 else ""
        sign_f1 = "+" if delta_f1 >= 0 else ""
        print(f"\n  SFT improvement over base:")
        print(f"    Macro EM Δ  {sign_em}{delta_em:.1%}")
        print(f"    Macro F1 Δ  {sign_f1}{delta_f1:.1%}")

    if "sft" in results and "dpo" in results:
        sft_f1 = macro_f1(results["sft"]["metrics"])
        dpo_f1 = macro_f1(results["dpo"]["metrics"])
        delta = dpo_f1 - sft_f1
        sign = "+" if delta >= 0 else ""
        print(f"\n  DPO vs SFT (macro F1 Δ):  {sign}{delta:.1%}", end="")
        if delta < 0:
            print("  ← alignment tax!", end="")
        print()

    if "teacher" in results:
        best_label = "dpo" if "dpo" in results else "sft" if "sft" in results else None
        if best_label:
            best_f1 = macro_f1(results[best_label]["metrics"])
            teacher_f1 = macro_f1(results["teacher"]["metrics"])
            gap = teacher_f1 - best_f1
            pct_closed = (1 - gap / teacher_f1) * 100 if teacher_f1 > 0 else 0
            print(f"\n  Gap to teacher ({best_label} macro F1): {gap:.1%}  ({pct_closed:.0f}% of teacher F1 reached)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",    default=str(DEFAULT_PATHS["base"]))
    parser.add_argument("--sft",     default=str(DEFAULT_PATHS["sft"]))
    parser.add_argument("--dpo",     default=str(DEFAULT_PATHS["dpo"]))
    parser.add_argument("--teacher", default=str(DEFAULT_PATHS["teacher"]))
    parser.add_argument("--output",  default=str(RESULTS_DIR / "comparison_summary.json"))
    args = parser.parse_args()

    results: dict[str, dict] = {}
    for label, path_str in [
        ("base", args.base), ("sft", args.sft),
        ("dpo", args.dpo), ("teacher", args.teacher),
    ]:
        r = load_result(Path(path_str))
        if r:
            results[label] = r
        else:
            print(f"  [{label}] not found: {path_str} — skipping")

    print_comparison(results)

    if results:
        summary = {
            label: {
                "model": r.get("model", "?"),
                "n": r["metrics"]["n"],
                "schema_validity_rate": r["metrics"]["schema_validity_rate"],
                "macro_em": round(macro_em(r["metrics"]), 4),
                "macro_f1": round(macro_f1(r["metrics"]), 4),
                "per_field_exact_match": r["metrics"]["per_field_exact_match"],
                "list_field_f1": r["metrics"]["list_field_f1"],
            }
            for label, r in results.items()
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"\nSummary → {args.output}")


if __name__ == "__main__":
    main()
