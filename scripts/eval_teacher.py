"""Evaluate GPT-4o-mini (teacher model) on the 200-example eval set.

Measures the "ceiling" we are trying to approach with SFT. Uses the same
TeacherClient and ExtractionResult pipeline as the distillation step so the
comparison is apples-to-apples.

Running this costs money (~$0.05 for 200 examples at GPT-4o-mini rates).
The script prints an estimate and asks for confirmation before making any calls.

Usage:
    python scripts/eval_teacher.py
    python scripts/eval_teacher.py --provider anthropic
    python scripts/eval_teacher.py --limit 20  # spot-check, ~$0.005
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from extractor.data.teacher import COST_PER_1M, TEACHER_MODELS, TeacherClient
from extractor.eval.metrics import eval_suite
from extractor.schemas.extraction import ExtractionResult
from extractor.utils.logging import configure_logging, get_logger

configure_logging("info")
logger = get_logger(__name__)

EVAL_PATH = Path("data/eval/human_audited.jsonl")
RESULTS_DIR = Path("data/eval")

# Rough token counts for cost estimate
AVG_INPUT_TOKENS = 800
AVG_OUTPUT_TOKENS = 200


def load_eval_set(limit: int | None) -> list[dict]:
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Eval set not found: {EVAL_PATH}")
    examples = [json.loads(line) for line in EVAL_PATH.read_text().splitlines() if line.strip()]
    return examples[:limit] if limit else examples


def estimate_cost(n: int, provider: str) -> float:
    rates = COST_PER_1M[provider]
    return n * (AVG_INPUT_TOKENS * rates["input"] + AVG_OUTPUT_TOKENS * rates["output"]) / 1e6


async def run_eval(provider: str, limit: int | None, concurrency: int) -> dict:
    examples = load_eval_set(limit)
    n = len(examples)
    model_id = TEACHER_MODELS[provider]

    cost_est = estimate_cost(n, provider)
    print(f"\nTeacher eval: {model_id}  ({n} examples)")
    print(f"Estimated cost: ~${cost_est:.3f}")
    confirm = input("Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return {}

    predictions: list[ExtractionResult] = []
    references: list[ExtractionResult] = []
    parse_errors: list[str | None] = []
    raw_results = []

    async with TeacherClient(provider, concurrency=concurrency) as client:
        for i, ex in enumerate(examples):
            pred, error, usage = await client.extract(ex["section_text"])
            ref = ExtractionResult.model_validate(ex["extraction"])

            predictions.append(pred)
            references.append(ref)
            parse_errors.append(error)

            if (i + 1) % 20 == 0:
                valid_so_far = sum(1 for e in parse_errors if e is None)
                logger.info(
                    "eval progress",
                    extra={"done": i + 1, "total": n, "valid": valid_so_far},
                )

            raw_results.append({
                "id": ex.get("id", i),
                "raw_response": pred.model_dump_json(),
                "parse_error": error,
                "prediction": pred.model_dump(),
                "reference": ref.model_dump(),
                "usage": usage,
            })

    metrics = eval_suite(predictions, references, parse_errors)
    logger.info("eval complete", extra=metrics)

    return {
        "label": "teacher",
        "model": model_id,
        "provider": provider,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "limit": limit,
        "metrics": metrics,
        "results": raw_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = asyncio.run(run_eval(args.provider, args.limit, args.concurrency))
    if not output:
        return

    metrics = output["metrics"]
    em = metrics["per_field_exact_match"]
    print(f"\n{output['model']} — {metrics['n']} examples")
    print(f"Schema validity: {metrics['schema_validity_rate']:.1%}")
    for field, score in em.items():
        print(f"  {field:<22} EM={score:.1%}")

    out_path = Path(args.output) if args.output else RESULTS_DIR / "teacher_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults → {out_path}")
    print("Next: python scripts/compare_evals.py")


if __name__ == "__main__":
    main()
