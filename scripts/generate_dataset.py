"""Teacher distillation: generate (section, extraction) training pairs.

Reads paper abstracts from data/raw/papers.jsonl, calls the teacher model,
validates output against the ExtractionResult schema, and writes valid examples
to data/raw/train_raw.jsonl.

Target: 2,000 validated examples.
Session 4: Generate first ~1,000. Session 5: Complete remainder + validate all.

Design decisions:
- Async with semaphore: generates ~5-10 examples in parallel, respecting rate limits.
- Checkpoint/resume: tracks completed arxiv_ids so crashes don't waste API budget.
- Validation at generation time: invalid outputs are logged and skipped, not saved.
  This means train_raw.jsonl contains only schema-valid examples — no garbage.

Usage:
    python scripts/generate_dataset.py --n 1000 --teacher openai
    python scripts/generate_dataset.py --n 1000 --teacher anthropic --resume
    python scripts/generate_dataset.py --n 2000 --resume  # resume after crash
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from extractor.data.teacher import Provider, TeacherClient
from extractor.utils.logging import configure_logging, get_logger

configure_logging("info")
logger = get_logger(__name__)

PAPERS_PATH = Path("data/raw/papers.jsonl")
OUT_PATH = Path("data/raw/train_raw.jsonl")
STATS_PATH = Path("data/raw/generation_stats.json")


def load_papers() -> list[dict]:
    if not PAPERS_PATH.exists():
        raise FileNotFoundError(
            f"{PAPERS_PATH} not found. Run `python scripts/fetch_papers.py` first."
        )
    papers = []
    for line in PAPERS_PATH.read_text().splitlines():
        if line.strip():
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return papers


def load_completed_ids(path: Path) -> set[str]:
    """Return set of arxiv_ids already written to the output file."""
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                ids.add(json.loads(line)["arxiv_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


async def process_paper(
    paper: dict,
    client: TeacherClient,
    out_file,  # open file handle
    lock: asyncio.Lock,
    counters: dict,
) -> None:
    """Generate and save one extraction example."""
    section_text = paper["abstract"]
    result, error, usage = await client.extract(section_text)

    async with lock:
        counters["total"] += 1

        if error:
            counters["invalid"] += 1
            logger.warning(
                "extraction failed",
                extra={
                    "arxiv_id": paper["arxiv_id"],
                    "error": error,
                    "total": counters["total"],
                },
            )
            return

        # The schema-valid example we save for training
        example = {
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "section_text": section_text,
            "extraction": result.model_dump(),
            "metadata": {
                "teacher_model": client.model,
                "provider": client.provider,
                "usage": usage,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        out_file.write(json.dumps(example) + "\n")
        out_file.flush()
        counters["valid"] += 1

        if counters["total"] % 50 == 0:
            logger.info(
                "progress",
                extra={
                    "total": counters["total"],
                    "valid": counters["valid"],
                    "invalid": counters["invalid"],
                    "validity_rate": round(
                        counters["valid"] / counters["total"], 3
                    ),
                    "cost_usd": round(client.usage.cost_usd(), 4),
                },
            )


async def run(provider: Provider, n: int, resume: bool, concurrency: int) -> None:
    papers = load_papers()
    logger.info("papers loaded", extra={"total": len(papers)})

    completed_ids = load_completed_ids(OUT_PATH) if resume else set()
    if resume and completed_ids:
        logger.info("resuming", extra={"already_done": len(completed_ids)})

    # Filter to papers we haven't processed yet
    todo = [p for p in papers if p["arxiv_id"] not in completed_ids][:n]
    logger.info("papers to process", extra={"count": len(todo)})

    if not todo:
        print("Nothing to do — all papers already processed. Use --n to increase target.")
        return

    # Rough cost estimate before spending money
    avg_input_tokens = 350   # typical abstract in tokens
    avg_output_tokens = 300  # typical JSON output in tokens
    from extractor.data.teacher import COST_PER_1M
    rates = COST_PER_1M[provider]
    estimated_cost = (
        len(todo) * avg_input_tokens / 1e6 * rates["input"]
        + len(todo) * avg_output_tokens / 1e6 * rates["output"]
    )
    print(f"\nProvider:  {provider}")
    print(f"Examples:  {len(todo)}")
    print(f"Est. cost: ${estimated_cost:.2f} USD")
    print(f"Output:    {OUT_PATH}")
    response = input("\nProceed? [y/N] ")
    if response.strip().lower() != "y":
        print("Aborted.")
        return

    counters: dict[str, int] = {"total": 0, "valid": 0, "invalid": 0}
    lock = asyncio.Lock()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"

    async with TeacherClient(provider, max_concurrency=concurrency) as client:
        with OUT_PATH.open(mode, encoding="utf-8") as fout:
            tasks = [
                process_paper(paper, client, fout, lock, counters)
                for paper in todo
            ]
            await asyncio.gather(*tasks)

    # Save generation stats
    stats = {
        "provider": provider,
        "model": TEACHER_MODELS[provider] if False else client.model,
        "n_requested": len(todo),
        "n_valid": counters["valid"],
        "n_invalid": counters["invalid"],
        "validity_rate": round(
            counters["valid"] / max(counters["total"], 1), 4
        ),
        "usage": client.usage.summary(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2))

    print(f"\n{'='*50}")
    print(f"Done.  Valid: {counters['valid']}  Invalid: {counters['invalid']}")
    print(f"Cost:  ${client.usage.cost_usd():.4f} USD")
    print(f"Stats: {STATS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher",
        choices=["openai", "anthropic"],
        default="openai",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of new examples to generate this run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip papers already in output file.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max simultaneous API calls.",
    )
    args = parser.parse_args()
    asyncio.run(run(args.teacher, args.n, args.resume, args.concurrency))


# Import needed for stats (fixes forward reference)
from extractor.data.teacher import TEACHER_MODELS  # noqa: E402

if __name__ == "__main__":
    main()
