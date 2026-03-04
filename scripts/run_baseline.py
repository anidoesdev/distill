"""Zero-shot baseline evaluation on 5 sample paper sections.

Runs the chosen model with no fine-tuning and records raw outputs and metrics.
This establishes the floor we need to beat after SFT (measured again in session 12).

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --model Qwen/Qwen2.5-1.5B-Instruct --load-in-4bit
    python scripts/run_baseline.py --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from extractor.eval.metrics import field_presence_rates, schema_validity_rate
from extractor.prompt import build_messages
from extractor.schemas.extraction import ExtractionResult


def run_baseline(model_name: str, load_in_4bit: bool) -> None:
    from extractor.model.inference import GenerationConfig, HFInference

    samples_path = Path("data/eval/sample_sections.json")
    samples: list[dict] = json.loads(samples_path.read_text())

    print(f"Model:   {model_name}")
    print(f"4-bit:   {load_in_4bit}")
    print(f"Samples: {len(samples)}\n")

    model = HFInference(model_name, load_in_4bit=load_in_4bit)
    gen_config = GenerationConfig(max_new_tokens=1024, temperature=0.0)

    results = []
    parse_errors: list[str | None] = []

    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {sample['paper_id']}: {sample['title'][:55]}...")

        messages = build_messages(sample["text"])
        response, meta = model.generate(messages, config=gen_config)
        extracted, error = ExtractionResult.from_model_output(response)
        parse_errors.append(error)

        status = "OK " if error is None else "ERR"
        presence = extracted.field_presence()
        filled = sum(presence.values())
        print(
            f"  [{status}] {filled}/6 fields present | "
            f"{meta['prompt_tokens']}→{meta['completion_tokens']} tok | "
            f"{meta['latency_s']}s"
        )
        if error:
            print(f"  Parse error: {error}")
            print(f"  Raw (first 200 chars): {response[:200]}")

        results.append(
            {
                "paper_id": sample["paper_id"],
                "title": sample["title"],
                "raw_response": response,
                "extracted": extracted.model_dump(),
                "parse_error": error,
                "field_presence": presence,
                "meta": meta,
                "reference": sample["expected"],
            }
        )

    # ── Summary metrics ───────────────────────────────────────────────────────
    extracted_objs = [ExtractionResult.model_validate(r["extracted"]) for r in results]
    validity = schema_validity_rate(parse_errors)
    presence = field_presence_rates(extracted_objs)

    print("\n" + "=" * 55)
    print("ZERO-SHOT BASELINE SUMMARY")
    print("=" * 55)
    print(f"Schema validity rate:  {validity:.0%}  ({sum(1 for e in parse_errors if e is None)}/{len(samples)})")
    print()
    print("Field presence rates (non-empty / total):")
    for field, rate in presence.items():
        bar = "#" * int(rate * 20)
        print(f"  {field:<20} {rate:.0%}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "model": model_name,
        "load_in_4bit": load_in_4bit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "schema_validity_rate": round(validity, 4),
            "field_presence_rates": presence,
        },
        "results": results,
    }
    out_path = Path("data/eval/baseline_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nFull results → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    run_baseline(args.model, args.load_in_4bit)


if __name__ == "__main__":
    main()
