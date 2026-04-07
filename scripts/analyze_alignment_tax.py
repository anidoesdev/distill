"""Alignment tax analysis: compare DPO model against SFT baseline field-by-field.

Reads two eval result JSONs and reports:
  - Per-field F1 delta (DPO − SFT)
  - Fields that improved, regressed, or stayed neutral
  - Whether the overall alignment tax is acceptable

An "alignment tax" is a regression on a task metric caused by the alignment
training. In our case: if DPO improves authors F1 but regresses methodology,
the methodology regression is the alignment tax.

Usage:
    python scripts/analyze_alignment_tax.py
    python scripts/analyze_alignment_tax.py \
        --baseline data/eval/sft_eval_results.json \
        --dpo      data/eval/dpo_eval_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extractor.eval.metrics import LIST_FIELDS, alignment_tax

RESULTS_DIR = Path("data/eval")


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("metrics")


def print_tax_report(tax: dict, baseline_label: str, dpo_label: str) -> None:
    deltas = tax["per_field_f1_delta"]
    em_deltas = tax["per_field_em_delta"]

    print("\n" + "=" * 60)
    print("ALIGNMENT TAX ANALYSIS")
    print("=" * 60)
    print(f"  Baseline: {baseline_label}")
    print(f"  DPO:      {dpo_label}")
    print()

    macro_f1_d = tax["macro_f1_delta"]
    macro_em_d = tax["macro_em_delta"]
    sign_f1 = "+" if macro_f1_d >= 0 else ""
    sign_em = "+" if macro_em_d >= 0 else ""
    print(f"  Macro F1 delta:  {sign_f1}{macro_f1_d:.1%}")
    print(f"  Macro EM delta:  {sign_em}{macro_em_d:.1%}")
    print()

    print(f"  {'Field':<22}  {'F1 Δ':>8}  {'EM Δ':>8}  Status")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  ------")
    for field in LIST_FIELDS:
        f1_d = deltas.get(field, 0.0)
        em_d = em_deltas.get(field, 0.0)
        if f1_d > 0.005:
            status = "improved"
        elif f1_d < -0.005:
            status = "REGRESSED  <--"
        else:
            status = "neutral"
        sign = "+" if f1_d >= 0 else ""
        em_sign = "+" if em_d >= 0 else ""
        print(f"  {field:<22}  {sign}{f1_d:>7.1%}  {em_sign}{em_d:>7.1%}  {status}")

    # methodology (str field — EM only)
    meth_em = em_deltas.get("methodology", 0.0)
    meth_sign = "+" if meth_em >= 0 else ""
    meth_status = "improved" if meth_em > 0.005 else ("REGRESSED  <--" if meth_em < -0.005 else "neutral")
    print(f"  {'methodology':<22}  {'n/a':>8}  {meth_sign}{meth_em:>7.1%}  {meth_status}")

    print()
    improved = tax["improved_fields"]
    regressed = tax["regressed_fields"]

    if regressed:
        print(f"  Alignment tax detected on: {', '.join(regressed)}")
        print()
        print("  Interpretation:")
        print("    A small regression (< 0.02) on 1-2 fields is normal and acceptable")
        print("    if the overall macro F1 improved. The preference data may not have")
        print("    covered these fields evenly.")
        print()
        print("  Options if tax is unacceptable:")
        print("    1. Re-generate preference pairs weighted toward regressed fields.")
        print("    2. Increase beta (reduces divergence from SFT reference).")
        print("    3. Accept SFT checkpoint and skip DPO for deployment.")
    else:
        print(f"  No alignment tax detected.")
        if improved:
            print(f"  DPO improved: {', '.join(improved)}")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=str(RESULTS_DIR / "sft_eval_results.json"))
    parser.add_argument("--dpo",      default=str(RESULTS_DIR / "dpo_eval_results.json"))
    parser.add_argument("--output",   default=str(RESULTS_DIR / "alignment_tax.json"))
    args = parser.parse_args()

    baseline_metrics = load_metrics(Path(args.baseline))
    dpo_metrics = load_metrics(Path(args.dpo))

    if not baseline_metrics:
        print(f"Baseline results not found: {args.baseline}")
        print("Run: python scripts/eval_sft.py --model-dir checkpoints/merged")
        return
    if not dpo_metrics:
        print(f"DPO results not found: {args.dpo}")
        print("Run: python scripts/eval_sft.py --model-dir checkpoints/dpo-merged \\")
        print("       --output data/eval/dpo_eval_results.json")
        return

    tax = alignment_tax(baseline_metrics, dpo_metrics)
    print_tax_report(tax, args.baseline, args.dpo)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(tax, indent=2))
    print(f"\nAlignment tax data → {args.output}")


if __name__ == "__main__":
    main()
