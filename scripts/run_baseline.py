"""Zero-shot baseline evaluation on 5 sample paper sections.

Runs the chosen model with no fine-tuning and records raw outputs.
This establishes the baseline we need to beat after SFT (session 12).

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --model Qwen/Qwen2.5-1.5B-Instruct --load-in-4bit
    python scripts/run_baseline.py --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a scientific information extractor. "
    "Given a section of a scientific paper, extract the following fields and return "
    "ONLY valid JSON with no additional text:\n\n"
    "{\n"
    '  "authors": ["list of author names"],\n'
    '  "methodology": "description of methods used",\n'
    '  "datasets_used": ["list of dataset names"],\n'
    '  "key_findings": ["list of key results"],\n'
    '  "limitations": ["list of stated limitations"],\n'
    '  "statistical_tests": ["list of statistical tests mentioned"]\n'
    "}\n\n"
    "Return empty lists or empty strings for fields not mentioned in the text. "
    "Do not infer information not explicitly stated."
)


def run_baseline(model_name: str, load_in_4bit: bool) -> None:
    from extractor.model.inference import GenerationConfig, HFInference

    samples_path = Path("data/eval/sample_sections.json")
    samples = json.loads(samples_path.read_text())

    print(f"Model: {model_name}")
    print(f"4-bit: {load_in_4bit}")
    print(f"Samples: {len(samples)}\n")

    model = HFInference(model_name, load_in_4bit=load_in_4bit)
    gen_config = GenerationConfig(max_new_tokens=1024, temperature=0.0)

    results = []
    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {sample['paper_id']}: {sample['title'][:50]}...")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["text"].strip()},
        ]

        response, meta = model.generate(messages, config=gen_config)

        # Try to parse JSON output
        parsed_output = None
        parse_error = None
        try:
            parsed_output = json.loads(response)
        except json.JSONDecodeError as e:
            parse_error = str(e)

        print(f"  Tokens: {meta['prompt_tokens']} in / {meta['completion_tokens']} out")
        print(f"  Latency: {meta['latency_s']}s  ({meta['tokens_per_s']} tok/s)")
        print(f"  Valid JSON: {'yes' if parsed_output else 'no'}")

        if parsed_output:
            # Quick field-presence check
            expected_keys = {"authors", "methodology", "datasets_used",
                             "key_findings", "limitations", "statistical_tests"}
            missing = expected_keys - set(parsed_output.keys())
            if missing:
                print(f"  Missing fields: {missing}")
        else:
            print(f"  Parse error: {parse_error}")
            print(f"  Raw output: {response[:200]}...")

        results.append({
            "paper_id": sample["paper_id"],
            "title": sample["title"],
            "raw_response": response,
            "parsed_output": parsed_output,
            "parse_error": parse_error,
            "meta": meta,
            "expected": sample["expected"],
        })

    # Save results
    out = {
        "model": model_name,
        "load_in_4bit": load_in_4bit,
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }
    out_path = Path("data/eval/baseline_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {out_path}")

    # Summary
    valid = sum(1 for r in results if r["parsed_output"])
    print(f"\nSummary: {valid}/{len(results)} valid JSON outputs")
    avg_tps = sum(r["meta"]["tokens_per_s"] for r in results) / len(results)
    print(f"Average: {avg_tps:.1f} tok/s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    run_baseline(args.model, args.load_in_4bit)


if __name__ == "__main__":
    main()
