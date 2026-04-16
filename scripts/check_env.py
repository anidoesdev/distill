"""Pre-flight environment check before any GPU training or eval run.

Verifies that all required packages, data files, and hardware are in the
expected state. Run this at the start of any session that touches the GPU
to catch problems before spending time downloading models.

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --phase dpo   # additional checks for DPO phase
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = "✓" if ok else "✗"
    line = f"  {icon} {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def run_checks(phase: str) -> int:
    failures = 0

    section("Python")
    v = sys.version_info
    ok = v >= (3, 10)
    if not check(f"Python >= 3.10", ok, f"found {v.major}.{v.minor}"):
        failures += 1

    section("Core packages")
    required = [
        ("torch",           "PyTorch"),
        ("transformers",    "Transformers"),
        ("peft",            "PEFT"),
        ("trl",             "TRL"),
        ("datasets",        "HuggingFace Datasets"),
        ("pydantic",        "Pydantic"),
        ("fastapi",         "FastAPI"),
        ("bitsandbytes",    "BitsAndBytes"),
    ]
    for pkg, name in required:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            check(name, True, ver)
        except ImportError:
            check(name, False, "not installed")
            failures += 1

    section("CUDA")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if not check("CUDA available", cuda_ok):
            failures += 1
        if cuda_ok:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            check(f"GPU: {device_name}", True, f"{vram_gb:.1f} GB VRAM")
            bf16_ok = torch.cuda.is_bf16_supported()
            if not check("BF16 supported (Ampere+)", bf16_ok, "required for training"):
                print("    → BF16 not available; set fp16=True, bf16=False in SFTConfig")
                failures += 1
    except ImportError:
        pass

    section("Data files")
    data_files = [
        ("data/eval/human_audited.jsonl", "Human-audited eval set (200 examples)"),
        ("data/processed/split_manifest.json", "Split manifest"),
    ]
    for path_str, label in data_files:
        p = Path(path_str)
        if p.exists():
            size = p.stat().st_size
            check(label, True, f"{size:,} bytes")
        else:
            check(label, False, f"missing: {path_str}")
            failures += 1

    hf_dirs = [
        ("data/processed/hf_dataset/train", "HF dataset / train split"),
        ("data/processed/hf_dataset/val",   "HF dataset / val split"),
    ]
    for path_str, label in hf_dirs:
        p = Path(path_str)
        check(label, p.exists(), path_str if not p.exists() else "")
        if not p.exists():
            failures += 1

    section("Checkpoints")
    ckpt_files = [
        ("checkpoints/sft",    "SFT adapter checkpoint"),
        ("checkpoints/merged", "Merged SFT model"),
    ]
    for path_str, label in ckpt_files:
        p = Path(path_str)
        check(label, p.exists(), "(optional — needed for eval/DPO)" if not p.exists() else "")

    section("Extractor package")
    try:
        import extractor
        check("extractor importable", True, f"v{extractor.__version__}")
    except ImportError as e:
        check("extractor importable", False, str(e))
        failures += 1

    try:
        from extractor.eval import eval_suite
        check("extractor.eval importable", True)
    except ImportError as e:
        check("extractor.eval importable", False, str(e))
        failures += 1

    try:
        from training.config import TrainingConfig
        cfg = TrainingConfig()
        check("training.config loads", True, f"model={cfg.model_name}")
    except Exception as e:
        check("training.config loads", False, str(e))
        failures += 1

    if phase == "dpo":
        section("DPO phase extras")
        dpo_packages = [
            ("trl",  "TRL DPOTrainer"),
        ]
        for pkg, name in dpo_packages:
            try:
                mod = importlib.import_module(pkg)
                has_dpo = hasattr(mod, "DPOTrainer")
                check(f"{name} available", has_dpo, getattr(mod, "__version__", "?"))
                if not has_dpo:
                    failures += 1
            except ImportError:
                check(name, False, "not installed")
                failures += 1

        pref_path = Path("data/processed/preference_pairs.jsonl")
        check(
            "Preference pairs dataset",
            pref_path.exists(),
            "(generated in session 16)" if not pref_path.exists() else f"{pref_path.stat().st_size:,} bytes",
        )

    if phase == "vllm":
        section("vLLM phase extras")

        # httpx required for the async client
        try:
            import httpx
            check("httpx", True, httpx.__version__)
        except ImportError:
            check("httpx", False, "pip install httpx")
            failures += 1

        # AWQ checkpoint (deployment target)
        awq_path = Path("checkpoints/awq")
        if not check("AWQ checkpoint", awq_path.exists(),
                     "(run quantize_awq.py)" if not awq_path.exists() else ""):
            failures += 1

        # vLLM reachability (non-fatal — vLLM may not be started yet)
        try:
            import urllib.request
            from extractor.config import settings
            url = f"{settings.vllm_base_url.rstrip('/v1')}/health"
            urllib.request.urlopen(url, timeout=3)
            check("vLLM server reachable", True, url)
        except Exception:
            # Not a failure — vLLM may be started separately
            check("vLLM server reachable", False, "start with: docker compose up vllm")

        # VLLMClient importable
        try:
            from extractor.model.vllm_client import VLLMClient  # noqa: F401
            check("VLLMClient importable", True)
        except ImportError as e:
            check("VLLMClient importable", False, str(e))
            failures += 1

    print()
    if failures == 0:
        print("All checks passed.")
    else:
        print(f"{failures} check(s) failed — fix before running training.")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="sft", choices=["sft", "dpo", "vllm"],
                        help="Phase to check extras for (default: sft)")
    args = parser.parse_args()
    failures = run_checks(args.phase)
    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
