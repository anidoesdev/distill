"""Update MODEL_CARD.md with actual benchmark numbers from saved eval results.

Reads the comparison summary and quantization benchmark JSONs and replaces
the placeholder (~XX%) values in MODEL_CARD.md with real numbers.

Run after completing sessions 20 (benchmark) and 18 (alignment tax eval):
    python scripts/update_model_card.py

Usage:
    python scripts/update_model_card.py
    python scripts/update_model_card.py --dry-run   # print changes without writing
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MODEL_CARD = Path("MODEL_CARD.md")
COMPARISON_JSON = Path("data/eval/comparison_summary.json")
BENCHMARK_JSON = Path("data/eval/quantization_benchmark.json")
ALIGNMENT_TAX_JSON = Path("data/eval/alignment_tax.json")


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def fmt_f1(val: float | None) -> str:
    return f"~{val:.0%}" if val is not None else "n/a"


def fmt_validity(val: float | None) -> str:
    return f"~{val:.0%}" if val is not None else "n/a"


def build_replacements(comparison: dict | None, benchmark: dict | None, tax: dict | None) -> list[tuple[str, str]]:
    """Return list of (pattern, replacement) pairs to apply to MODEL_CARD.md."""
    replacements = []

    if comparison:
        for label, display in [("base", "base"), ("sft", "SFT"), ("dpo", "DPO"), ("awq", "AWQ"), ("teacher", "teacher")]:
            r = comparison.get(label)
            if not r:
                continue
            f1 = r.get("macro_f1")
            validity = r.get("schema_validity_rate")
            if f1:
                # Replace the placeholder in the model comparison table
                if label == "base":
                    replacements.append((r"base\) \| ~\d+%", f"base) | {fmt_f1(f1)}"))
                elif label == "teacher":
                    replacements.append((r"teacher\) \| ~\d+%", f"teacher) | {fmt_f1(f1)}"))
                elif label == "sft":
                    replacements.append((r"\*\*EXTRACTOR SFT\*\* \| \*\*~\d+%\*\*", f"**EXTRACTOR SFT** | **{fmt_f1(f1)}**"))
                elif label == "dpo":
                    replacements.append((r"\*\*EXTRACTOR DPO\*\* \| \*\*~\d+%\*\*", f"**EXTRACTOR DPO** | **{fmt_f1(f1)}**"))
                elif label == "awq":
                    replacements.append((r"\*\*EXTRACTOR AWQ\*\* \| \*\*~\d+%\*\*", f"**EXTRACTOR AWQ** | **{fmt_f1(f1)}**"))

    if benchmark and "awq" in benchmark:
        awq = benchmark["awq"]
        tps = awq.get("avg_tokens_per_sec")
        latency = awq.get("avg_latency_s")
        vram = awq.get("vram_reserved_gb")
        size = awq.get("size_gb")
        if tps:
            replacements.append((r"~60 tokens/sec", f"~{tps:.0f} tokens/sec"))
        if latency:
            response_time = 200 / tps if tps else None
            if response_time:
                replacements.append((r"~3\.5 seconds", f"~{response_time:.1f} seconds"))
        if vram:
            replacements.append((r"~1\.7 GB\n\| Model size", f"~{vram:.1f} GB\n| Model size"))
        if size:
            replacements.append((r"~1\.5 GB\n\| Cost", f"~{size:.1f} GB\n| Cost"))

    return replacements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not MODEL_CARD.exists():
        print(f"MODEL_CARD.md not found: {MODEL_CARD}")
        return

    comparison = load_json(COMPARISON_JSON)
    benchmark = load_json(BENCHMARK_JSON)
    tax = load_json(ALIGNMENT_TAX_JSON)

    if not any([comparison, benchmark]):
        print("No eval results found. Run eval scripts first.")
        print(f"  Expected: {COMPARISON_JSON}")
        print(f"  Expected: {BENCHMARK_JSON}")
        return

    content = MODEL_CARD.read_text()
    replacements = build_replacements(comparison, benchmark, tax)

    if not replacements:
        print("No placeholders to update.")
        return

    n_changed = 0
    for pattern, replacement in replacements:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            n_changed += 1
            content = new_content

    if args.dry_run:
        print(f"Would update {n_changed} values in MODEL_CARD.md")
        for p, r in replacements:
            print(f"  {p!r} → {r!r}")
    else:
        MODEL_CARD.write_text(content)
        print(f"MODEL_CARD.md updated ({n_changed} values replaced).")
        print("Review with: git diff MODEL_CARD.md")


if __name__ == "__main__":
    main()
