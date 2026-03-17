"""Merge LoRA adapters into the base model and save a standalone model.

After SFT training, the checkpoint directory contains only the LoRA adapter
weights (~50 MB), not the full model. This script:
  1. Loads the base model in full precision (bf16)
  2. Loads the LoRA adapters on top via PeftModel
  3. Merges adapter weights into the base matrices (merge_and_unload)
  4. Saves the merged model + tokenizer to a new directory

The merged model can then be:
  - Loaded with AutoModelForCausalLM.from_pretrained (no PEFT dependency)
  - Quantized with llama.cpp / GGUF for local inference
  - Uploaded to HuggingFace Hub

Usage:
    python scripts/export_checkpoint.py
    python scripts/export_checkpoint.py --adapter checkpoints/sft --output checkpoints/merged
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from extractor.utils.logging import configure_logging, get_logger
from training.config import TrainingConfig

configure_logging("info")
logger = get_logger(__name__)


def export(adapter_dir: str, output_dir: str, base_model: str) -> None:
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_path}\n"
            "Run full training first: python training/train_sft.py"
        )

    logger.info("loading tokenizer", extra={"model": base_model})
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load base in bf16 — no quantization, we need real weights for merging.
    # Quantized (4-bit) weights cannot be merged; merge must happen at fp16/bf16.
    logger.info("loading base model in bf16", extra={"model": base_model})
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("loading LoRA adapters", extra={"adapter_dir": str(adapter_path)})
    model = PeftModel.from_pretrained(base, str(adapter_path))

    # merge_and_unload folds the LoRA matrices (B @ A * alpha/r) into the base
    # weight matrices in-place, then returns a plain transformers model with no
    # PEFT wrapper. After this, model is identical to a model trained from scratch.
    logger.info("merging adapters into base weights")
    model = model.merge_and_unload()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("saving merged model", extra={"output_dir": str(out)})
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    logger.info("export complete", extra={"output_dir": str(out), "size_mb": round(size_mb, 1)})
    print(f"\nMerged model saved to: {out}  ({size_mb:.0f} MB)")
    print("Load with:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{out}')")


def main() -> None:
    cfg = TrainingConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default=cfg.output_dir,
        help=f"Path to LoRA adapter checkpoint (default: {cfg.output_dir})",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/merged",
        help="Output directory for merged model (default: checkpoints/merged)",
    )
    parser.add_argument(
        "--base-model",
        default=cfg.model_name,
        help=f"Base model name or path (default: {cfg.model_name})",
    )
    args = parser.parse_args()

    export(args.adapter, args.output, args.base_model)


if __name__ == "__main__":
    main()
