"""Generate 1,000 preference pairs for DPO training.

Reads from the existing SFT training data (teacher-distilled examples) and
produces (prompt, chosen, rejected) triples in three ways:

  1. degraded (70%): teacher output as chosen, programmatically degraded as rejected
  2. base_model (20%): teacher output as chosen, base model eval output as rejected
     (only for examples where the base model produced valid but incorrect JSON)
  3. partial (10%): full teacher output as chosen, teacher output with one field
     cleared as rejected (very clear preference signal)

Output: data/processed/preference_pairs.jsonl

Each line is a JSON object matching the PreferencePair schema. The prompt
field uses the model's chat template format so it feeds directly to DPOTrainer.

Usage:
    python scripts/generate_preferences.py
    python scripts/generate_preferences.py --n 500 --seed 99
    python scripts/generate_preferences.py --base-eval data/eval/base_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from extractor.data.degrade import degrade, degrade_composite
from extractor.data.splits import load_split
from extractor.data.tokenize import format_assistant_response
from extractor.prompt import build_messages
from extractor.schemas.preference import PreferencePair
from extractor.utils.logging import configure_logging, get_logger
from training.config import TrainingConfig

configure_logging("info")
logger = get_logger(__name__)

OUTPUT_PATH = Path("data/processed/preference_pairs.jsonl")

# Target distribution
STRATEGY_COUNTS = {
    "degraded": 0.70,
    "base_model": 0.20,
    "partial": 0.10,
}


def make_prompt(section_text: str, tokenizer) -> str:
    """Format the prompt using the model's chat template.

    The prompt ends with the assistant turn opener so DPOTrainer can concatenate
    chosen/rejected completions directly.
    """
    messages = build_messages(section_text)
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def generate_degraded(examples: list[dict], n: int, tokenizer, rng: random.Random) -> list[dict]:
    """Chosen = teacher output, rejected = programmatically degraded."""
    selected = rng.choices(examples, k=n)
    pairs = []
    for ex in selected:
        extraction = ex["extraction"]
        chosen_str = format_assistant_response(extraction)

        # Mix single and composite degradation: 60/40
        if rng.random() < 0.6:
            degraded, strategy = degrade(extraction, rng)
            strategy_tag = strategy
        else:
            degraded, strategies = degrade_composite(extraction, rng, n=2)
            strategy_tag = "+".join(strategies)

        rejected_str = format_assistant_response(degraded)

        # Skip if degradation produced no change (e.g., field was already empty)
        if chosen_str == rejected_str:
            continue

        pair = PreferencePair(
            id=f"{ex.get('arxiv_id', 'unknown')}_{rng.randint(0, 999999):06d}",
            prompt=make_prompt(ex["section_text"], tokenizer),
            chosen=chosen_str,
            rejected=rejected_str,
            source="degraded",
        )
        pairs.append(pair.model_dump())
    return pairs


def generate_from_base_model(
    examples: list[dict],
    base_eval_path: str,
    n: int,
    tokenizer,
    rng: random.Random,
) -> list[dict]:
    """Chosen = teacher output, rejected = base model output from session 13 eval.

    Only uses examples where the base model produced valid JSON with at least
    one non-empty field — random garbage isn't a useful rejected signal.
    """
    eval_path = Path(base_eval_path)
    if not eval_path.exists():
        logger.warning("base eval results not found, skipping base_model strategy",
                       extra={"path": base_eval_path})
        return []

    base_results = json.loads(eval_path.read_text()).get("results", [])
    # Build index by section_text hash for matching
    base_by_id = {r.get("id", i): r for i, r in enumerate(base_results)}

    # Find training examples that appear in the base eval (by position)
    usable = []
    for i, ex in enumerate(examples):
        base_r = base_by_id.get(i)
        if not base_r:
            continue
        if base_r.get("parse_error"):
            continue
        base_pred = base_r.get("prediction", {})
        teacher_str = format_assistant_response(ex["extraction"])
        base_str = format_assistant_response(base_pred)
        if teacher_str == base_str:
            continue
        usable.append((ex, base_str))

    if not usable:
        logger.warning("no usable base_model pairs found")
        return []

    selected = rng.choices(usable, k=min(n, len(usable)))
    pairs = []
    for ex, base_str in selected:
        pair = PreferencePair(
            id=f"{ex.get('arxiv_id', 'unknown')}_base_{rng.randint(0, 999999):06d}",
            prompt=make_prompt(ex["section_text"], tokenizer),
            chosen=format_assistant_response(ex["extraction"]),
            rejected=base_str,
            source="base_model",
        )
        pairs.append(pair.model_dump())
    return pairs


def generate_partial(examples: list[dict], n: int, tokenizer, rng: random.Random) -> list[dict]:
    """Chosen = full teacher output, rejected = same with one field cleared.

    This is the strongest preference signal: the only difference is one missing field.
    """
    from extractor.data.degrade import clear_field

    CLEARABLE = ["datasets_used", "key_findings", "limitations", "statistical_tests"]
    selected = rng.choices(examples, k=n)
    pairs = []
    for ex in selected:
        extraction = ex["extraction"]
        # Only useful if the field we clear is non-empty
        non_empty = [f for f in CLEARABLE if extraction.get(f)]
        if not non_empty:
            continue
        field = rng.choice(non_empty)
        rejected_extraction = clear_field(extraction, field)
        pair = PreferencePair(
            id=f"{ex.get('arxiv_id', 'unknown')}_partial_{rng.randint(0, 999999):06d}",
            prompt=make_prompt(ex["section_text"], tokenizer),
            chosen=format_assistant_response(extraction),
            rejected=format_assistant_response(rejected_extraction),
            source="partial",
        )
        pairs.append(pair.model_dump())
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Total pairs to generate (default: 1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-eval", default="data/eval/base_eval_results.json")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cfg = TrainingConfig()

    logger.info("loading tokenizer for prompt formatting", extra={"model": cfg.model_name})
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

    logger.info("loading training split")
    train_examples = load_split("train")
    logger.info("training examples loaded", extra={"n": len(train_examples)})

    n_degraded   = int(args.n * STRATEGY_COUNTS["degraded"])
    n_base_model = int(args.n * STRATEGY_COUNTS["base_model"])
    n_partial    = args.n - n_degraded - n_base_model

    logger.info(
        "generating preference pairs",
        extra={"n_degraded": n_degraded, "n_base_model": n_base_model, "n_partial": n_partial},
    )

    all_pairs: list[dict] = []

    degraded_pairs = generate_degraded(train_examples, n_degraded, tokenizer, rng)
    logger.info("degraded pairs generated", extra={"n": len(degraded_pairs)})
    all_pairs.extend(degraded_pairs)

    base_pairs = generate_from_base_model(train_examples, args.base_eval, n_base_model, tokenizer, rng)
    logger.info("base_model pairs generated", extra={"n": len(base_pairs)})
    all_pairs.extend(base_pairs)

    partial_pairs = generate_partial(train_examples, n_partial, tokenizer, rng)
    logger.info("partial pairs generated", extra={"n": len(partial_pairs)})
    all_pairs.extend(partial_pairs)

    # Shuffle so strategies are interleaved during training
    rng.shuffle(all_pairs)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(p) for p in all_pairs))

    # Summary
    sources = {}
    for p in all_pairs:
        sources[p["source"]] = sources.get(p["source"], 0) + 1

    print(f"\nPreference pairs generated: {len(all_pairs)}")
    print(f"Output: {out}")
    print("\nBreakdown by source:")
    for src, count in sorted(sources.items()):
        print(f"  {src:<15} {count:>5}  ({count/len(all_pairs):.0%})")

    # Spot check: print one example of each source
    print("\nSample (first of each source):")
    seen = set()
    for p in all_pairs:
        if p["source"] not in seen:
            seen.add(p["source"])
            chosen_preview = p["chosen"][:80].replace("\n", " ")
            rejected_preview = p["rejected"][:80].replace("\n", " ")
            print(f"\n  source: {p['source']}")
            print(f"  chosen:   {chosen_preview}...")
            print(f"  rejected: {rejected_preview}...")


if __name__ == "__main__":
    main()
