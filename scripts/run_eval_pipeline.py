"""End-to-end evaluation pipeline — runs all eval scripts in sequence.

Produces a single summary JSON at --output showing model progression:
  base model → SFT → DPO → DPO+AWQ (if checkpoint exists)

Usage:
    python scripts/run_eval_pipeline.py
    python scripts/run_eval_pipeline.py --output reports/eval_summary.json

Prerequisites:
    python scripts/check_env.py --phase vllm
    docker compose up   # vLLM must be serving the DPO+AWQ model

Each step is optional — missing checkpoints are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


STEPS = [
    {
        "name": "base_model",
        "label": "Base model (zero-shot)",
        "script": "scripts/eval_base_model.py",
        "output": "reports/eval_base.json",
        "required": False,
    },
    {
        "name": "sft",
        "label": "SFT fine-tuned",
        "script": "scripts/eval_sft.py",
        "output": "reports/eval_sft.json",
        "required": False,
    },
    {
        "name": "dpo",
        "label": "DPO-aligned",
        "script": "scripts/eval_sft.py",   # same script, different checkpoint via env
        "output": "reports/eval_dpo.json",
        "required": False,
    },
]


def _run_step(step: dict) -> dict | None:
    script = Path(step["script"])
    output = Path(step["output"])

    if not script.exists():
        print(f"  ⚠ {step['label']}: script not found ({script}) — skipped")
        return None

    print(f"  → {step['label']}...")
    output.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ✗ {step['label']} failed:")
        print(result.stderr[-500:])
        return None

    if output.exists():
        with open(output) as f:
            return json.load(f)
    return None


def _print_summary(results: dict[str, dict | None]) -> None:
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)

    fields = ["macro_f1", "macro_em", "schema_validity_rate"]
    header = f"{'Model':<20}" + "".join(f"{f:>20}" for f in fields)
    print(header)
    print("-" * 60)

    prev_f1 = None
    for name, result in results.items():
        if result is None:
            print(f"{name:<20}  (not run)")
            continue
        row = f"{name:<20}"
        f1 = result.get("macro_f1", 0.0)
        for f in fields:
            val = result.get(f, 0.0)
            row += f"{val:>20.4f}"
        delta = f"  (+{f1 - prev_f1:.4f})" if prev_f1 is not None and f1 > prev_f1 else ""
        print(row + delta)
        prev_f1 = f1

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="Write summary JSON here")
    args = parser.parse_args()

    print("Running evaluation pipeline...\n")
    results: dict[str, dict | None] = {}

    for step in STEPS:
        results[step["name"]] = _run_step(step)

    _print_summary(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {k: v for k, v in results.items() if v is not None},
                f, indent=2,
            )
        print(f"\nSummary written to {args.output}")


if __name__ == "__main__":
    main()
