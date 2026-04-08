"""Quantize the merged DPO model to 4-bit AWQ format.

AWQ (Activation-aware Weight Quantization) reduces model size from ~6 GB (BF16)
to ~1.5 GB while preserving quality by protecting the most salient weight channels.

Process:
  1. Load merged model in BF16
  2. Run 128 calibration examples to detect salient channels
  3. Scale salient channels to preserve them through quantization
  4. Quantize all weights to INT4 with per-group zero-point
  5. Save quantized model (compatible with AutoModelForCausalLM)

The AWQ model can be served directly with vLLM (session 22) or Transformers.

Requirements:
    pip install autoawq

Usage:
    python scripts/quantize_awq.py
    python scripts/quantize_awq.py --model-dir checkpoints/dpo-merged
    python scripts/quantize_awq.py --calib-samples 64  # faster, slightly less accurate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extractor.utils.logging import configure_logging, get_logger
from training.config import QuantizationConfig

configure_logging("info")
logger = get_logger(__name__)


def load_calibration_data(dataset_dir: str, n_samples: int, seq_len: int) -> list[str]:
    """Load calibration text from the training split.

    AWQ calibration only needs raw text — no labels. We use the section_text
    field from our training examples since it matches the model's input distribution.
    """
    from datasets import load_from_disk

    ds = load_from_disk(dataset_dir)
    texts = []
    for ex in ds.select(range(min(n_samples, len(ds)))):
        # Each example has a 'messages' column with [system, user, assistant]
        # Use the user message content (the section text) for calibration
        messages = ex.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                texts.append(msg["content"])
                break
    logger.info("calibration data loaded", extra={"n": len(texts)})
    return texts


def quantize(cfg: QuantizationConfig) -> None:
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise ImportError(
            "autoawq not installed. Run:\n"
            "  pip install autoawq\n"
            "Note: autoawq requires CUDA and a compatible GPU."
        )

    from transformers import AutoTokenizer

    model_dir = Path(cfg.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Merged model not found: {model_dir}\n"
            "Export first: python scripts/export_checkpoint.py"
        )

    output_dir = Path(cfg.awq_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading tokenizer", extra={"model_dir": str(model_dir)})
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    logger.info("loading model for AWQ calibration", extra={"model_dir": str(model_dir)})
    model = AutoAWQForCausalLM.from_pretrained(
        str(model_dir),
        low_cpu_mem_usage=True,
        use_cache=False,
    )

    quant_config = {
        "zero_point": cfg.awq_zero_point,
        "q_group_size": cfg.awq_group_size,
        "w_bit": cfg.awq_bits,
        "version": "GEMM",   # GEMM kernel is faster than GEMV for batch inference
    }
    logger.info("AWQ quantization config", extra=quant_config)

    calib_data = load_calibration_data(
        cfg.awq_calib_data,
        cfg.awq_calib_samples,
        cfg.awq_calib_seq_len,
    )

    logger.info("running AWQ calibration and quantization")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    logger.info("saving AWQ model", extra={"output_dir": str(output_dir)})
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Record what we did for session 20 benchmark
    meta = {
        "source_model": str(model_dir),
        "method": "awq",
        "bits": cfg.awq_bits,
        "group_size": cfg.awq_group_size,
        "zero_point": cfg.awq_zero_point,
        "calib_samples": cfg.awq_calib_samples,
    }
    (output_dir / "quantization_meta.json").write_text(json.dumps(meta, indent=2))

    size_gb = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / 1e9
    logger.info("AWQ quantization complete", extra={"output_dir": str(output_dir), "size_gb": round(size_gb, 2)})
    print(f"\nAWQ model saved to: {output_dir}  ({size_gb:.1f} GB)")
    print("Load with:")
    print("  from awq import AutoAWQForCausalLM")
    print(f"  model = AutoAWQForCausalLM.from_quantized('{output_dir}')")
    print("\nOr with vLLM (session 22):")
    print(f"  vllm serve {output_dir} --quantization awq")


def main() -> None:
    cfg = QuantizationConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=cfg.model_dir)
    parser.add_argument("--output",    default=cfg.awq_output_dir)
    parser.add_argument("--calib-samples", type=int, default=cfg.awq_calib_samples)
    parser.add_argument("--bits",      type=int, default=cfg.awq_bits, choices=[4, 8])
    args = parser.parse_args()

    cfg = cfg.model_copy(update={
        "model_dir": args.model_dir,
        "awq_output_dir": args.output,
        "awq_calib_samples": args.calib_samples,
        "awq_bits": args.bits,
    })
    quantize(cfg)


if __name__ == "__main__":
    main()
