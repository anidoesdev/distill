"""Convert the merged model to GGUF format for llama.cpp inference.

GGUF (GPT-Generated Unified Format) is llama.cpp's binary format. It enables:
  - CPU inference (no GPU required)
  - Mixed CPU+GPU offloading (--n-gpu-layers N)
  - Fast tokenization and sampling built into llama.cpp
  - Ollama, LM Studio, and other local inference tools

Process:
  1. Clone llama.cpp (one-time setup)
  2. Run convert-hf-to-gguf.py to produce an F16 GGUF
  3. Run llama-quantize to produce the final Q4_K_M (or other) GGUF

Requirements:
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && make -j$(nproc)          # or cmake on Windows
    pip install -r llama.cpp/requirements/requirements-convert-hf-to-gguf.txt

Usage:
    python scripts/quantize_gguf.py
    python scripts/quantize_gguf.py --model-dir checkpoints/dpo-merged
    python scripts/quantize_gguf.py --type Q5_K_M
    python scripts/quantize_gguf.py --llama-cpp-dir /path/to/llama.cpp
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from extractor.utils.logging import configure_logging, get_logger
from training.config import QuantizationConfig

configure_logging("info")
logger = get_logger(__name__)


def check_llama_cpp(llama_dir: Path) -> None:
    """Verify llama.cpp is cloned and the convert script exists."""
    convert_script = llama_dir / "convert_hf_to_gguf.py"
    # newer llama.cpp uses convert_hf_to_gguf.py; older uses convert-hf-to-gguf.py
    if not convert_script.exists():
        convert_script = llama_dir / "convert-hf-to-gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"llama.cpp convert script not found in {llama_dir}\n"
            "Clone and build llama.cpp:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  cd llama.cpp && make -j$(nproc)\n"
            "  pip install -r requirements/requirements-convert-hf-to-gguf.txt"
        )

    quantize_bin = llama_dir / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_dir / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        raise FileNotFoundError(
            f"llama-quantize binary not found. Build llama.cpp:\n"
            "  cd llama.cpp && make -j$(nproc)"
        )


def convert_to_f16_gguf(model_dir: Path, output_dir: Path, llama_dir: Path) -> Path:
    """Step 1: convert HuggingFace model to F16 GGUF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    f16_path = output_dir / "model-f16.gguf"

    convert_script = llama_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        convert_script = llama_dir / "convert-hf-to-gguf.py"

    logger.info("converting to F16 GGUF", extra={"script": str(convert_script)})
    cmd = [
        sys.executable, str(convert_script),
        str(model_dir),
        "--outfile", str(f16_path),
        "--outtype", "f16",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if result.stdout:
        logger.info("convert stdout", extra={"output": result.stdout[-500:]})

    logger.info("F16 GGUF written", extra={"path": str(f16_path)})
    return f16_path


def quantize_gguf(f16_path: Path, output_dir: Path, quant_type: str, llama_dir: Path) -> Path:
    """Step 2: quantize F16 GGUF to the target type (e.g. Q4_K_M)."""
    quantize_bin = llama_dir / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_dir / "build" / "bin" / "llama-quantize"

    out_path = output_dir / f"model-{quant_type}.gguf"

    logger.info("quantizing GGUF", extra={"type": quant_type, "output": str(out_path)})
    cmd = [str(quantize_bin), str(f16_path), str(out_path), quant_type]
    subprocess.run(cmd, check=True)

    size_gb = out_path.stat().st_size / 1e9
    logger.info("GGUF quantization complete", extra={"path": str(out_path), "size_gb": round(size_gb, 2)})
    return out_path


def convert_and_quantize(cfg: QuantizationConfig) -> None:
    model_dir = Path(cfg.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Merged model not found: {model_dir}\n"
            "Export first: python scripts/export_checkpoint.py"
        )

    llama_dir = Path(cfg.llama_cpp_dir)
    check_llama_cpp(llama_dir)

    output_dir = Path(cfg.gguf_output_dir)

    f16_path = convert_to_f16_gguf(model_dir, output_dir, llama_dir)
    quant_path = quantize_gguf(f16_path, output_dir, cfg.gguf_type, llama_dir)

    size_gb = quant_path.stat().st_size / 1e9
    print(f"\nGGUF model saved: {quant_path}  ({size_gb:.1f} GB)")
    print("\nRun with llama.cpp:")
    print(f"  ./llama.cpp/llama-cli -m {quant_path} -p 'Extract structured info...' -n 512")
    print("\nOr with Ollama:")
    print(f"  ollama create extractor -f Modelfile   # Modelfile FROM {quant_path}")


def main() -> None:
    cfg = QuantizationConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",    default=cfg.model_dir)
    parser.add_argument("--output",       default=cfg.gguf_output_dir)
    parser.add_argument("--type",         default=cfg.gguf_type,
                        choices=["Q4_K_M", "Q5_K_M", "Q8_0", "Q4_0"])
    parser.add_argument("--llama-cpp-dir", default=cfg.llama_cpp_dir)
    args = parser.parse_args()

    cfg = cfg.model_copy(update={
        "model_dir": args.model_dir,
        "gguf_output_dir": args.output,
        "gguf_type": args.type,
        "llama_cpp_dir": args.llama_cpp_dir,
    })
    convert_and_quantize(cfg)


if __name__ == "__main__":
    main()
