"""Prepare HuggingFace Dataset from JSONL splits for SFT training.

Converts train_split.jsonl and val_split.jsonl to HF Dataset format and
saves to disk. Also runs sequence length analysis to inform max_length choice.

The saved datasets have a single "messages" column (list of role/content dicts).
TRL SFTTrainer consumes this directly with DataCollatorForCompletionOnlyLM
to apply the chat template and mask prompt tokens in the loss.

Output layout:
  data/processed/hf_dataset/
    train/       — HF Dataset, arrow format
    val/         — HF Dataset, arrow format
    length_report.json

Usage:
    python scripts/prepare_dataset.py --model Qwen/Qwen2.5-3B-Instruct
    python scripts/prepare_dataset.py --model Qwen/Qwen2.5-3B-Instruct --n-samples 200
"""

import argparse
import json
from pathlib import Path

from extractor.data.splits import load_split, verify_no_leakage
from extractor.data.tokenize import (
    compute_token_counts,
    example_to_messages,
    print_length_report,
    sequence_length_report,
)
from extractor.utils.logging import configure_logging, get_logger

configure_logging("info")
logger = get_logger(__name__)

HF_DATASET_DIR = Path("data/processed/hf_dataset")
LENGTH_REPORT_PATH = Path("data/processed/length_report.json")


def build_hf_dataset(examples: list[dict], split_name: str):
    """Convert a list of training examples to a HuggingFace Dataset.

    Each row has one field: "messages" — the full conversation including the
    assistant response. TRL will apply the chat template at training time.
    """
    from datasets import Dataset

    rows = [{"messages": example_to_messages(ex)} for ex in examples]
    ds = Dataset.from_list(rows)
    logger.info(f"built hf dataset", extra={"split": split_name, "rows": len(ds)})
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID for tokenizer (used only for length analysis).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of examples to tokenize for length analysis. Use 0 for all.",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip tokenization and length analysis (faster, no GPU/model needed).",
    )
    args = parser.parse_args()

    # ── Leakage check ─────────────────────────────────────────────────────────
    verify_no_leakage()
    logger.info("leakage check passed")

    # ── Load splits ───────────────────────────────────────────────────────────
    train_examples = load_split("train")
    val_examples = load_split("val")
    logger.info("splits loaded", extra={"train": len(train_examples), "val": len(val_examples)})

    # ── Sequence length analysis ──────────────────────────────────────────────
    length_report = {}
    if not args.no_analysis:
        print(f"\nLoading tokenizer: {args.model}")
        print("(This downloads the tokenizer if not cached — ~50MB)")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        n_samples = args.n_samples or len(train_examples)
        print(f"Tokenizing {n_samples} train examples for length analysis...")

        counts = compute_token_counts(train_examples, tokenizer, n_samples=n_samples)
        length_report = sequence_length_report(counts)
        print_length_report(length_report, counts)

        LENGTH_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        LENGTH_REPORT_PATH.write_text(json.dumps(length_report, indent=2))
        print(f"\nReport saved to {LENGTH_REPORT_PATH}")
    else:
        print("Skipping length analysis (--no-analysis)")

    # ── Build and save HF Datasets ────────────────────────────────────────────
    print("\nBuilding HF Datasets...")
    train_ds = build_hf_dataset(train_examples, "train")
    val_ds = build_hf_dataset(val_examples, "val")

    HF_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(HF_DATASET_DIR / "train"))
    val_ds.save_to_disk(str(HF_DATASET_DIR / "val"))

    print(f"\nSaved to {HF_DATASET_DIR}/")
    print(f"  train: {len(train_ds)} examples")
    print(f"  val:   {len(val_ds)} examples")

    # ── Verify structure ──────────────────────────────────────────────────────
    sample = train_ds[0]
    print("\nExample structure (first train example):")
    for msg in sample["messages"]:
        content_preview = msg["content"][:80].replace("\n", " ")
        print(f"  [{msg['role']:<12}] {content_preview}...")

    if length_report:
        rec_p = length_report.get("p95", 1024)
        import math
        rec = 2 ** math.ceil(math.log2(rec_p + 100))
        rec = max(512, min(rec, 2048))
        print(f"\nSet max_seq_length={rec} in SFTTrainer (session 9).")
        print("Set packing=True for efficient training.")


if __name__ == "__main__":
    main()
