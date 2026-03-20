"""Evaluate the fine-tuned SFT model on the 200-example human-audited eval set.

Runs inference on every example in data/eval/human_audited.jsonl, computes
structured accuracy metrics against human-verified references, and writes a
full results file for use in the session 13 model comparison.

Usage:
    python scripts/eval_sft.py
    python scripts/eval_sft.py --model-dir checkpoints/merged
    python scripts/eval_sft.py --model-dir checkpoints/sft --adapter
    python scripts/eval_sft.py --limit 20   # quick spot-check, not full eval
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from extractor.eval.metrics import eval_suite
from extractor.prompt import build_messages
from extractor.schemas.extraction import ExtractionResult
from extractor.utils.logging import configure_logging, get_logger

configure_logging("info")
logger = get_logger(__name__)

EVAL_PATH = Path("data/eval/human_audited.jsonl")
RESULTS_DIR = Path("data/eval")


def load_eval_set(limit: int | None = None) -> list[dict]:
    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Eval set not found: {EVAL_PATH}\n"
            "Run the audit tool first: streamlit run scripts/audit_app.py"
        )
    examples = [json.loads(line) for line in EVAL_PATH.read_text().splitlines() if line.strip()]
    if limit:
        examples = examples[:limit]
    return examples


def run_eval(model_dir: str, use_adapter: bool, limit: int | None) -> dict:
    """Load model, run inference on eval set, compute metrics."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.config import TrainingConfig

    cfg = TrainingConfig()

    logger.info("loading tokenizer", extra={"model_dir": model_dir})
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if use_adapter:
        from peft import PeftModel
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("loading base model in 4-bit for adapter inference")
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        logger.info("loading merged model in bf16", extra={"model_dir": model_dir})
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
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

        # Slice off the prompt tokens — output_ids includes input
        response_ids = output_ids[0][input_ids.shape[1]:]
        raw = tokenizer.decode(response_ids, skip_special_tokens=True)

        pred, error = ExtractionResult.from_model_output(raw)
        ref = ExtractionResult.model_validate(ex["extraction"])

        predictions.append(pred)
        references.append(ref)
        parse_errors.append(error)

        if (i + 1) % 20 == 0:
            valid_so_far = sum(1 for e in parse_errors if e is None)
            logger.info(
                "eval progress",
                extra={"done": i + 1, "total": len(examples), "valid": valid_so_far},
            )

        raw_results.append({
            "id": ex.get("id", i),
            "section_text_preview": ex["section_text"][:120],
            "raw_response": raw,
            "parse_error": error,
            "prediction": pred.model_dump(),
            "reference": ref.model_dump(),
        })

    metrics = eval_suite(predictions, references, parse_errors)
    logger.info("eval complete", extra=metrics)

    return {
        "model_dir": model_dir,
        "use_adapter": use_adapter,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "limit": limit,
        "metrics": metrics,
        "results": raw_results,
    }


def print_report(output: dict) -> None:
    metrics = output["metrics"]
    em = metrics["per_field_exact_match"]
    f1s = metrics["list_field_f1"]

    print("\n" + "=" * 62)
    print("SFT EVAL REPORT")
    print("=" * 62)
    print(f"  Model:             {output['model_dir']}")
    print(f"  Examples:          {metrics['n']}")
    print(f"  Schema validity:   {metrics['schema_validity_rate']:.1%}")
    print()
    print(f"  {'Field':<22} {'ExactMatch':>10}  {'F1':>7}  {'P':>7}  {'R':>7}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*7}")
    for field in ["authors", "methodology", "datasets_used", "key_findings", "limitations", "statistical_tests"]:
        exact = em.get(field, 0.0)
        f1_row = f1s.get(field, {})
        f1 = f1_row.get("f1", "-")
        p  = f1_row.get("precision", "-")
        r  = f1_row.get("recall", "-")
        if isinstance(f1, float):
            print(f"  {field:<22} {exact:>10.1%}  {f1:>7.1%}  {p:>7.1%}  {r:>7.1%}")
        else:
            print(f"  {field:<22} {exact:>10.1%}  {'n/a':>7}  {'n/a':>7}  {'n/a':>7}")
    print("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="checkpoints/merged",
        help="Path to merged model or adapter checkpoint (default: checkpoints/merged)",
    )
    parser.add_argument(
        "--adapter",
        action="store_true",
        help="Load model_dir as a LoRA adapter on top of the base model. "
             "Use when you haven't run export_checkpoint.py yet.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N examples (default: all 200)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (default: data/eval/sft_eval_results.json)",
    )
    args = parser.parse_args()

    output = run_eval(args.model_dir, args.adapter, args.limit)
    print_report(output)

    out_path = Path(args.output) if args.output else RESULTS_DIR / "sft_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nFull results → {out_path}")
    print("\nNext: python scripts/compare_models.py (session 13 — SFT vs base vs GPT-4o-mini)")


if __name__ == "__main__":
    main()
