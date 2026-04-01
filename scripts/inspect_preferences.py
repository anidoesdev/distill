"""Inspect the generated preference dataset before DPO training.

Reports:
  - Total pairs and source breakdown
  - Average prompt/chosen/rejected token lengths
  - F1 score distribution between chosen and rejected
  - Examples with suspiciously small margins

Usage:
    python scripts/inspect_preferences.py
    python scripts/inspect_preferences.py --path data/processed/preference_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extractor.eval.metrics import list_field_f1, per_field_exact_match
from extractor.schemas.extraction import ExtractionResult
from extractor.utils.logging import configure_logging

configure_logging("info")

PREF_PATH = Path("data/processed/preference_pairs.jsonl")


def token_len_approx(text: str) -> int:
    """Rough token count: ~4 chars per token for English/JSON."""
    return len(text) // 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(PREF_PATH))
    parser.add_argument("--show-small-margin", type=int, default=5,
                        help="Show N examples with smallest chosen-rejected F1 gap")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Dataset not found: {path}")
        print("Run: python scripts/generate_preferences.py")
        return

    pairs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    n = len(pairs)

    print(f"\nPreference dataset: {path}")
    print(f"Total pairs: {n}")

    # Source breakdown
    sources: dict[str, int] = {}
    for p in pairs:
        src = p.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print("\nSource breakdown:")
    for src, count in sorted(sources.items()):
        print(f"  {src:<16} {count:>5}  ({count/n:.0%})")

    # Token length stats
    prompt_lens = [token_len_approx(p["prompt"]) for p in pairs]
    chosen_lens = [token_len_approx(p["chosen"]) for p in pairs]
    rejected_lens = [token_len_approx(p["rejected"]) for p in pairs]

    def stats(vals: list[int]) -> str:
        s = sorted(vals)
        return f"min={s[0]}  p50={s[len(s)//2]}  p95={s[int(len(s)*0.95)]}  max={s[-1]}"

    print("\nApprox token lengths (chars ÷ 4):")
    print(f"  prompt:   {stats(prompt_lens)}")
    print(f"  chosen:   {stats(chosen_lens)}")
    print(f"  rejected: {stats(rejected_lens)}")

    # Parse chosen/rejected and compute F1 gap
    margins = []
    parse_failures = 0
    for p in pairs:
        chosen_result, err_c = ExtractionResult.from_model_output(p["chosen"])
        rejected_result, err_r = ExtractionResult.from_model_output(p["rejected"])
        if err_c or err_r:
            parse_failures += 1
            continue
        # Use key_findings F1 as a proxy for overall quality
        chosen_f1 = list_field_f1([chosen_result], [rejected_result], "key_findings")["f1"]
        margins.append((chosen_f1, p))

    print(f"\nParse failures: {parse_failures}/{n}")

    if margins:
        margin_vals = sorted([m for m, _ in margins])
        small = [(m, p) for m, p in sorted(margins, key=lambda x: x[0])[:args.show_small_margin]]
        print(f"\nChosen vs rejected F1 gap (key_findings):")
        print(f"  min={margin_vals[0]:.3f}  p25={margin_vals[len(margin_vals)//4]:.3f}  "
              f"p50={margin_vals[len(margin_vals)//2]:.3f}  max={margin_vals[-1]:.3f}")

        if small:
            print(f"\nSmallest-margin examples (potential noise):")
            for margin, p in small:
                print(f"  source={p['source']:<12} margin={margin:.3f}")
                print(f"    chosen:   {p['chosen'][:70]}...")
                print(f"    rejected: {p['rejected'][:70]}...")


if __name__ == "__main__":
    main()
