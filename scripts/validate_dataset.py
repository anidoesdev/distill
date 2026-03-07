"""Validate and clean the distilled training dataset.

Reads data/raw/train_raw.jsonl, runs all validators, and writes:
  data/processed/train_clean.jsonl   — examples passing all checks
  data/processed/train_rejected.jsonl — examples with failure reasons

Also prints a distribution report so you can catch systematic quality issues
before they become training problems.

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --raw data/raw/train_raw.jsonl
    python scripts/validate_dataset.py --strict  # also reject cross-ref failures
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from extractor.data.validate import ALL_VALIDATORS, run_validators
from extractor.schemas.extraction import ExtractionResult
from extractor.utils.logging import configure_logging, get_logger

configure_logging("info")
logger = get_logger(__name__)

RAW_PATH = Path("data/raw/train_raw.jsonl")
CLEAN_PATH = Path("data/processed/train_clean.jsonl")
REJECTED_PATH = Path("data/processed/train_rejected.jsonl")
REPORT_PATH = Path("data/processed/validation_report.json")

FIELDS = [
    "authors", "methodology", "datasets_used",
    "key_findings", "limitations", "statistical_tests",
]


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("skipping malformed line", extra={"error": str(e)})
    return examples


def field_presence_summary(examples: list[dict]) -> dict[str, float]:
    """Fraction of examples where each field is non-empty."""
    counts: dict[str, int] = {f: 0 for f in FIELDS}
    for ex in examples:
        result = ExtractionResult.model_validate(ex.get("extraction", {}))
        for field, present in result.field_presence().items():
            if present:
                counts[field] += 1
    n = len(examples)
    return {f: round(counts[f] / n, 3) for f in FIELDS}


def length_distribution(examples: list[dict]) -> dict[str, dict]:
    """For each field, report mean and p50/p90 value lengths."""
    import statistics

    field_lengths: dict[str, list[int]] = defaultdict(list)
    for ex in examples:
        result = ExtractionResult.model_validate(ex.get("extraction", {}))
        for field in FIELDS:
            val = getattr(result, field)
            if isinstance(val, str):
                field_lengths[field].append(len(val.split()))
            elif isinstance(val, list):
                field_lengths[field].append(len(val))

    stats = {}
    for field, lengths in field_lengths.items():
        if not lengths:
            continue
        sorted_l = sorted(lengths)
        n = len(sorted_l)
        stats[field] = {
            "mean": round(statistics.mean(lengths), 1),
            "p50": sorted_l[n // 2],
            "p90": sorted_l[int(n * 0.90)],
            "max": max(lengths),
        }
    return stats


def failure_reason_summary(rejected: list[dict]) -> dict[str, int]:
    """Count how many examples failed each validator prefix."""
    prefix_counts: Counter = Counter()
    for ex in rejected:
        for reason in ex.get("failure_reasons", []):
            prefix = reason.split(":")[0]
            prefix_counts[prefix] += 1
    return dict(prefix_counts.most_common())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(RAW_PATH))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reject examples with any failure, including soft cross-ref failures.",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run generate_dataset.py first.")
        return

    print(f"Loading {raw_path}...")
    examples = load_jsonl(raw_path)
    print(f"Loaded {len(examples)} raw examples\n")

    # Check for duplicate arxiv_ids
    id_counts: Counter = Counter(ex.get("arxiv_id", "") for ex in examples)
    duplicates = {k: v for k, v in id_counts.items() if v > 1 and k}
    if duplicates:
        print(f"WARNING: {len(duplicates)} duplicate arxiv_ids found. Keeping first occurrence.")

    seen_ids: set[str] = set()
    clean: list[dict] = []
    rejected: list[dict] = []

    for ex in examples:
        arxiv_id = ex.get("arxiv_id", "")

        # Deduplication
        if arxiv_id and arxiv_id in seen_ids:
            rejected.append({**ex, "failure_reasons": ["dedup: duplicate arxiv_id"]})
            continue
        if arxiv_id:
            seen_ids.add(arxiv_id)

        reasons = run_validators(ex)

        # In non-strict mode, crossref failures are warnings, not rejections
        if not args.strict:
            reasons = [r for r in reasons if not r.startswith("crossref:")]

        if reasons:
            rejected.append({**ex, "failure_reasons": reasons})
        else:
            clean.append(ex)

    # Write outputs
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)

    CLEAN_PATH.write_text(
        "\n".join(json.dumps(ex) for ex in clean) + "\n",
        encoding="utf-8",
    )
    REJECTED_PATH.write_text(
        "\n".join(json.dumps(ex) for ex in rejected) + "\n",
        encoding="utf-8",
    )

    # Distribution analysis on clean set
    presence = field_presence_summary(clean)
    lengths = length_distribution(clean)
    failures = failure_reason_summary(rejected)

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(f"  Raw examples:      {len(examples)}")
    print(f"  Duplicates:        {len(examples) - len(seen_ids) - (len(examples) - len(seen_ids)):>4}")
    print(f"  Clean (training):  {len(clean):>4}  ({len(clean)/len(examples):.0%})")
    print(f"  Rejected:          {len(rejected):>4}  ({len(rejected)/len(examples):.0%})")

    print("\nField presence in clean set:")
    for field, rate in presence.items():
        bar = "#" * int(rate * 25)
        print(f"  {field:<22} {rate:.0%}  {bar}")

    print("\nField length distribution (words for str, count for list):")
    for field, stat in lengths.items():
        print(
            f"  {field:<22} mean={stat['mean']:5.1f}  "
            f"p50={stat['p50']:3d}  p90={stat['p90']:3d}  max={stat['max']:3d}"
        )

    if failures:
        print("\nFailure reasons (rejected examples):")
        for reason, count in failures.items():
            print(f"  {reason:<25} {count}")

    print(f"\nClean dataset: {CLEAN_PATH}")
    print(f"Rejected:      {REJECTED_PATH}")

    # Save machine-readable report
    report = {
        "raw_count": len(examples),
        "clean_count": len(clean),
        "rejected_count": len(rejected),
        "pass_rate": round(len(clean) / len(examples), 4),
        "field_presence": presence,
        "field_lengths": lengths,
        "failure_reasons": failures,
        "duplicate_ids": len(duplicates),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Report:        {REPORT_PATH}")

    # Warn if clean dataset is below minimum viable size
    if len(clean) < 1500:
        print(
            f"\nWARNING: Clean dataset has only {len(clean)} examples. "
            "Target is 2,000. Consider generating more with "
            "`python scripts/generate_dataset.py --resume --n 2500`."
        )


if __name__ == "__main__":
    main()
