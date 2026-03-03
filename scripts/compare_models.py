"""Model comparison script — run before committing to a base model.

Loads each candidate, runs a single extraction prompt, and reports:
  - Parameter count
  - VRAM used after loading
  - Time to first token (TTFT) and tokens/sec
  - Raw model output (quality assessed visually)

Usage:
    python scripts/compare_models.py                  # all candidates
    python scripts/compare_models.py --model qwen3b   # single model
    python scripts/compare_models.py --load-in-4bit   # use 4-bit quantization
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

CANDIDATES = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b":   "Qwen/Qwen2.5-3B-Instruct",
    "llama3b":  "meta-llama/Llama-3.2-3B-Instruct",
    "phi3.5":   "microsoft/Phi-3.5-mini-instruct",
}

SAMPLE_TEXT = """
We trained a convolutional neural network on ImageNet (1.2 million images, 1000 classes)
using stochastic gradient descent with momentum 0.9, weight decay 5e-4, and initial
learning rate 0.01 decayed by 10x at epochs 30 and 60. Top-1 accuracy reached 73.4%
and top-5 accuracy reached 91.2%. Data augmentation included random crops, horizontal
flips, and color jitter. Dropout with p=0.5 was applied before the final classifier.
Authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton.
Limitation: model was not evaluated on out-of-distribution datasets.
Chi-squared test confirmed class imbalance was not a confound (p < 0.001).
"""

SYSTEM_PROMPT = (
    "You are a scientific information extractor. "
    "Extract structured information from the provided text and return ONLY valid JSON "
    "matching this schema: "
    "{\"authors\": [str], \"methodology\": str, \"datasets_used\": [str], "
    "\"key_findings\": [str], \"limitations\": [str], \"statistical_tests\": [str]}"
)


def run_one(model_key: str, model_name: str, load_in_4bit: bool) -> dict:
    from extractor.model.inference import GenerationConfig, HFInference

    print(f"\n{'='*60}")
    print(f"  {model_key}: {model_name}")
    print(f"{'='*60}")

    vram_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    t_load = time.perf_counter()
    model = HFInference(model_name, load_in_4bit=load_in_4bit)
    load_time = time.perf_counter() - t_load

    vram_after = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    vram_used = vram_after - vram_before

    print(f"Load time: {load_time:.1f}s  |  VRAM delta: {vram_used:.2f} GB")

    # Inspect chat template so we can see what format the model expects
    model.inspect_chat_template()

    # Show how the model tokenizes scientific notation
    model.tokenize_example("p < 0.001 and R² = 0.94")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SAMPLE_TEXT.strip()},
    ]

    response, meta = model.generate(messages, config=GenerationConfig(max_new_tokens=512))

    print(f"\nOutput:\n{response}")
    print(f"\nMeta: {meta}")

    # Check if output is valid JSON
    is_valid_json = False
    try:
        parsed = json.loads(response)
        is_valid_json = True
        print(f"Valid JSON: yes  |  Keys: {list(parsed.keys())}")
    except json.JSONDecodeError as e:
        print(f"Valid JSON: no  |  Error: {e}")

    return {
        "model_key": model_key,
        "model_name": model_name,
        "load_time_s": round(load_time, 1),
        "vram_gb": round(vram_used, 2),
        "prompt_tokens": meta["prompt_tokens"],
        "completion_tokens": meta["completion_tokens"],
        "latency_s": meta["latency_s"],
        "tokens_per_s": meta["tokens_per_s"],
        "valid_json": is_valid_json,
        "response": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(CANDIDATES), default=None,
                        help="Run a single model. Omit to run all.")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    to_run = {args.model: CANDIDATES[args.model]} if args.model else CANDIDATES
    results = []

    for key, name in to_run.items():
        try:
            r = run_one(key, name, args.load_in_4bit)
            results.append(r)
        except Exception as exc:
            print(f"FAILED {key}: {exc}", file=sys.stderr)

    # Summary table
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<12} {'VRAM GB':>8} {'Load s':>7} {'tok/s':>7} {'JSON':>6}")
    print("-"*70)
    for r in results:
        print(
            f"{r['model_key']:<12} {r['vram_gb']:>8.2f} {r['load_time_s']:>7.1f} "
            f"{r['tokens_per_s']:>7.1f} {'yes' if r['valid_json'] else 'no':>6}"
        )

    out_path = Path("data/eval/model_comparison.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
