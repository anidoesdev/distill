"""Utilities for loading the train/val/eval splits into training scripts.

Used by training scripts in sessions 9-11 and eval scripts in sessions 12-13.
The manifest is the canonical source of truth for which examples are in each split.
"""

from __future__ import annotations

import json
from pathlib import Path

TRAIN_PATH = Path("data/processed/train_split.jsonl")
VAL_PATH = Path("data/processed/val_split.jsonl")
EVAL_PATH = Path("data/eval/human_audited.jsonl")
MANIFEST_PATH = Path("data/processed/split_manifest.json")


def load_split(split: str) -> list[dict]:
    """Load a named split. split must be 'train', 'val', or 'eval'.

    Returns list of example dicts, each with 'section_text' and 'extraction' keys.
    """
    paths = {"train": TRAIN_PATH, "val": VAL_PATH, "eval": EVAL_PATH}
    if split not in paths:
        raise ValueError(f"split must be one of {list(paths)}; got {split!r}")

    path = paths[split]
    if not path.exists():
        if split in ("train", "val"):
            raise FileNotFoundError(
                f"{path} not found. Run `python scripts/create_splits.py` first."
            )
        else:
            raise FileNotFoundError(
                f"{path} not found. Complete the audit tool (session 6) first."
            )

    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            examples.append(json.loads(line))
    return examples


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found. Run `python scripts/create_splits.py` first."
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def verify_no_leakage() -> None:
    """Assert no arxiv_id appears in both train/val and eval.

    Call this at the start of any training script to catch accidental leakage
    if data files are regenerated without re-running create_splits.py.
    """
    manifest = load_manifest()
    train_ids = set(manifest["splits"]["train"])
    val_ids = set(manifest["splits"]["val"])
    eval_ids = set(manifest["splits"]["eval"])

    train_val_overlap = train_ids & val_ids
    train_eval_overlap = train_ids & eval_ids
    val_eval_overlap = val_ids & eval_ids

    errors = []
    if train_val_overlap:
        errors.append(f"train/val overlap: {len(train_val_overlap)} examples")
    if train_eval_overlap:
        errors.append(f"train/eval overlap: {len(train_eval_overlap)} examples")
    if val_eval_overlap:
        errors.append(f"val/eval overlap: {len(val_eval_overlap)} examples")

    if errors:
        raise RuntimeError(
            "DATA LEAKAGE DETECTED — do not train with this split:\n"
            + "\n".join(f"  {e}" for e in errors)
        )


def split_stats() -> dict:
    manifest = load_manifest()
    return manifest["stats"]
