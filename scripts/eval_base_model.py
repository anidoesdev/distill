"""Evaluate the base model (no fine-tuning) on the 200-example eval set.

Provides the "before SFT" baseline for the session 13 comparison. Uses the
same eval machinery as eval_sft.py so results are directly comparable.

The base model runs in 4-bit NF4 to match training memory conditions.

Usage:
    python scripts/eval_base_model.py
    python scripts/eval_base_model.py --model Qwen/Qwen2.5-3B-Instruct
    python scripts/eval_base_model.py --limit 20
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from extractor.eval.metrics import eval_suite
from extractor.prompt import build_messages
from extractor.schemas.extraction import ExtractionResult
from extractor.utils.logging import configure_logging, get_logger

configure_logging("info")
logger = get_logger(__name__)

EVAL_PATH = Path("data/eval/human_audited.jsonl")
RESULTS_DIR = Path("data/eval")


def load_eval_set(limit: int | None) -> list[dict]:
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Eval set not found: {EVAL_PATH}")
    examples = [json.loads(line) for line in EVAL_PATH.read_text().splitlines() if line.strip()]
    return examples[:limit] if limit else examples


def run_eval(model_name: str, limit: int | None) -> dict:
    logger.info("loading tokenizer", extra={"model": model_name})
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info("loading base model in 4-bit NF4", extra={"model": model_name})
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    examples = load_eval_set(limit)
    logger.info("eval set loaded", extra={"n": len(examples)})

    predictions: list[ExtractionResult] = []
    references: list[ExtractionResult] = []
    parse_errors: list[str | None] = []
    raw_results = []

    for i, ex in enumerate(examples):
        messages = build_messages(ex["section_text"])
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response_ids = output_ids[0][input_ids.shape[1]:]
        raw = tokenizer.decode(response_ids, skip_special_tokens=True)

        pred, error = ExtractionResult.from_model_output(raw)
        ref = ExtractionResult.model_validate(ex["extraction"])

        predictions.append(pred)
        references.append(ref)
        parse_errors.append(error)

        if (i + 1) % 20 == 0:
            valid_so_far = sum(1 for e in parse_errors if e is None)
            logger.info("eval progress", extra={"done": i + 1, "total": len(examples), "valid": valid_so_far})

        raw_results.append({
            "id": ex.get("id", i),
            "raw_response": raw,
            "parse_error": error,
            "prediction": pred.model_dump(),
            "reference": ref.model_dump(),
        })

    metrics = eval_suite(predictions, references, parse_errors)
    logger.info("eval complete", extra=metrics)

    return {
        "label": "base",
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "limit": limit,
        "metrics": metrics,
        "results": raw_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = run_eval(args.model, args.limit)

    metrics = output["metrics"]
    em = metrics["per_field_exact_match"]
    print(f"\nBase model ({args.model}) — {metrics['n']} examples")
    print(f"Schema validity: {metrics['schema_validity_rate']:.1%}")
    for field, score in em.items():
        print(f"  {field:<22} EM={score:.1%}")

    out_path = Path(args.output) if args.output else RESULTS_DIR / "base_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
