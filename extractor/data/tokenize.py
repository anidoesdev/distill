"""Tokenization utilities for dataset preparation.

Converts JSONL training examples into the HuggingFace messages format consumed
by TRL SFTTrainer. Also provides sequence-length analysis so max_length can
be set empirically rather than guessed.

The messages format:
    [
        {"role": "system",    "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user",      "content": "Extract structured information...\n\n{text}"},
        {"role": "assistant", "content": '{"authors": [...], ...}'},
    ]

TRL's DataCollatorForCompletionOnlyLM masks loss on everything before the
assistant turn so only the JSON response contributes to training loss.
"""

from __future__ import annotations

import json
import statistics
from typing import Any

from extractor.prompt import build_messages


def format_assistant_response(extraction: dict[str, Any]) -> str:
    """Serialize the extraction dict to the compact JSON string the model should output.

    Using compact JSON (no indent) because:
    1. Fewer tokens = shorter sequences = more examples per packed batch.
    2. We want the fine-tuned model to output compact JSON for lower latency.
    Any consistent format works; what matters is that training and inference
    use the exact same format.
    """
    return json.dumps(extraction, ensure_ascii=False, separators=(",", ":"))


def example_to_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    """Convert one training example to the messages list format.

    The assistant turn is the target JSON the model should learn to produce.
    """
    messages = build_messages(example["section_text"])
    messages.append(
        {
            "role": "assistant",
            "content": format_assistant_response(example["extraction"]),
        }
    )
    return messages


def compute_token_counts(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    n_samples: int | None = None,
) -> list[int]:
    """Tokenize a sample of examples and return list of token counts.

    Args:
        examples: Training examples with 'section_text' and 'extraction'.
        tokenizer: HuggingFace tokenizer with apply_chat_template.
        n_samples: If set, sample this many examples (faster analysis).

    Returns:
        List of integers — one token count per example.
    """
    import random
    pool = random.Random(42).sample(examples, min(n_samples or len(examples), len(examples)))

    counts = []
    for ex in pool:
        messages = example_to_messages(ex)
        # apply_chat_template returns the full formatted string
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        ids = tokenizer.encode(formatted, add_special_tokens=False)
        counts.append(len(ids))

    return counts


def sequence_length_report(counts: list[int]) -> dict[str, Any]:
    """Compute percentile statistics for a list of token counts."""
    sorted_counts = sorted(counts)
    n = len(sorted_counts)

    def percentile(p: float) -> int:
        idx = max(0, min(int(p / 100 * n), n - 1))
        return sorted_counts[idx]

    report = {
        "n": n,
        "mean": round(statistics.mean(counts), 1),
        "std": round(statistics.stdev(counts) if n > 1 else 0.0, 1),
        "min": sorted_counts[0],
        "p50": percentile(50),
        "p75": percentile(75),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": sorted_counts[-1],
    }

    # How many examples exceed common max_length choices?
    for max_len in (512, 768, 1024, 1536, 2048):
        n_truncated = sum(1 for c in counts if c > max_len)
        report[f"truncated_at_{max_len}"] = round(n_truncated / n, 4)

    return report


def print_length_report(report: dict[str, Any], counts: list[int]) -> None:
    """Print a human-readable sequence length report with an ASCII histogram."""
    print("\n" + "=" * 55)
    print("SEQUENCE LENGTH ANALYSIS")
    print("=" * 55)
    print(f"  Examples analyzed:  {report['n']}")
    print(f"  Mean ± std:         {report['mean']:.0f} ± {report['std']:.0f} tokens")
    print(f"  Min / Max:          {report['min']} / {report['max']}")
    print(f"  p50 / p90 / p95:    {report['p50']} / {report['p90']} / {report['p95']}")

    print("\nTruncation at different max_length values:")
    for max_len in (512, 768, 1024, 1536, 2048):
        rate = report[f"truncated_at_{max_len}"]
        bar = "!" * int(rate * 30)
        print(f"  max_length={max_len:<5}  {rate:5.1%} truncated  {bar}")

    print("\nLength distribution (ASCII histogram):")
    # Bucket into 10 bins
    min_c, max_c = min(counts), max(counts)
    bucket_size = max(1, (max_c - min_c) // 10)
    buckets: dict[int, int] = {}
    for c in counts:
        bucket = ((c - min_c) // bucket_size) * bucket_size + min_c
        buckets[bucket] = buckets.get(bucket, 0) + 1
    max_count = max(buckets.values())
    for bucket in sorted(buckets):
        bar_len = int(buckets[bucket] / max_count * 30)
        bar = "#" * bar_len
        print(f"  {bucket:>5}-{bucket+bucket_size:<5} | {bar} ({buckets[bucket]})")

    # Recommendation
    rec = report["p95"] + 100  # 100-token buffer above p95
    # Round up to next power of 2
    import math
    rec = 2 ** math.ceil(math.log2(rec))
    rec = max(512, min(rec, 2048))
    print(f"\nRecommended max_length: {rec}  (covers {1 - report.get(f'truncated_at_{rec}', 0):.0%} of examples)")
    print("Use packing=True in SFTTrainer for efficient training.")
