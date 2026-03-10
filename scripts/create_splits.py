"""Create train/val split from the validated training dataset.

Excludes the 200 human-audited examples which are the permanent eval set.
Writes:
  data/processed/train_split.jsonl  — used for SFT (sessions 9-11)
  data/processed/val_split.jsonl    — used for early stopping
  data/processed/split_manifest.json — committed to git; records exact split

The manifest records the arxiv_id assigned to each split so the split can
be verified independently even if the data files are regenerated.

Usage:
    python scripts/create_splits.py
    python scripts/create_splits.py --val-ratio 0.15 --seed 42
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

CLEAN_PATH = Path("data/processed/train_clean.jsonl")
AUDITED_PATH = Path("data/eval/human_audited.jsonl")
TRAIN_PATH = Path("data/processed/train_split.jsonl")
VAL_PATH = Path("data/processed/val_split.jsonl")
MANIFEST_PATH = Path("data/processed/split_manifest.json")


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return examples


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(ex) for ex in examples) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not CLEAN_PATH.exists():
        print(f"ERROR: {CLEAN_PATH} not found. Run validate_dataset.py first.")
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    all_examples = load_jsonl(CLEAN_PATH)
    print(f"Clean examples loaded: {len(all_examples)}")

    audited_ids: set[str] = set()
    if AUDITED_PATH.exists():
        audited = load_jsonl(AUDITED_PATH)
        audited_ids = {ex["arxiv_id"] for ex in audited if ex.get("arxiv_id")}
        print(f"Audited (eval) IDs excluded: {len(audited_ids)}")
    else:
        print(
            f"WARNING: {AUDITED_PATH} not found. "
            "Run the audit tool (session 6) before splitting to prevent eval leakage."
        )

    # Check for leakage — fail loudly
    clean_ids = {ex["arxiv_id"] for ex in all_examples}
    leaked = audited_ids & clean_ids
    if leaked and AUDITED_PATH.exists():
        print(f"Overlap between clean and audited sets: {len(leaked)} examples.")
        print("These will be removed from the training pool.")

    # ── Build training pool ────────────────────────────────────────────────────
    pool = [ex for ex in all_examples if ex["arxiv_id"] not in audited_ids]
    print(f"Training pool after exclusions: {len(pool)}")

    if len(pool) < 100:
        print("ERROR: Too few examples. Generate more data before splitting.")
        return

    # ── Shuffle with fixed seed ───────────────────────────────────────────────
    # Use an isolated Random instance, not random.seed(), to avoid affecting
    # other code that depends on the global RNG.
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    # ── Split ─────────────────────────────────────────────────────────────────
    n_val = max(50, int(len(pool) * args.val_ratio))
    val_examples = pool[:n_val]
    train_examples = pool[n_val:]

    print(f"\nSplit (seed={args.seed}, val_ratio={args.val_ratio}):")
    print(f"  Train:  {len(train_examples)}")
    print(f"  Val:    {len(val_examples)}")
    print(f"  Eval:   {len(audited_ids)}  (human-audited, never trained on)")

    # ── Write data files ──────────────────────────────────────────────────────
    write_jsonl(TRAIN_PATH, train_examples)
    write_jsonl(VAL_PATH, val_examples)
    print(f"\nWrote {TRAIN_PATH}")
    print(f"Wrote {VAL_PATH}")

    # ── Write manifest ────────────────────────────────────────────────────────
    # The manifest is the reproducibility artifact. Commit this file.
    # It records exactly which arxiv_ids went into each split so the split
    # can be verified without storing the full data in git.
    manifest = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "total_clean": len(all_examples),
            "excluded_eval": len(audited_ids),
            "training_pool": len(pool),
            "train": len(train_examples),
            "val": len(val_examples),
        },
        "splits": {
            "train": sorted(ex["arxiv_id"] for ex in train_examples),
            "val": sorted(ex["arxiv_id"] for ex in val_examples),
            "eval": sorted(audited_ids),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH}  ← commit this file")

    # ── Category distribution ─────────────────────────────────────────────────
    from collections import Counter
    def top_cats(examples: list[dict], n: int = 5) -> list[tuple[str, int]]:
        cats: Counter = Counter()
        for ex in examples:
            for cat in ex.get("categories", [])[:1]:  # primary category only
                cats[cat] += 1
        return cats.most_common(n)

    print("\nCategory distribution (train):")
    for cat, count in top_cats(train_examples):
        print(f"  {cat:<20} {count}")


if __name__ == "__main__":
    main()
