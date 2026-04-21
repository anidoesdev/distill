"""End-to-end ingestion pipeline: paper sections → structured JSONL.

Reads a JSONL file where each line has at minimum:
    {"paper_id": "arxiv:2309.12345", "section_text": "We trained BERT on..."}

Calls the EXTRACTOR batch API and writes an output JSONL where each line
is the input record with an "extraction" key added:
    {
      "paper_id": "arxiv:2309.12345",
      "section_text": "...",
      "extraction": {
        "authors": [...],
        "methodology": "...",
        ...
      },
      "parse_error": null,
      "ingest_latency_s": 0.412
    }

Failed extractions (parse_error != null) are written to a separate
--failed JSONL for review.

Usage:
    python scripts/ingest_papers.py \\
        --input  data/raw/papers.jsonl \\
        --output data/processed/papers_extracted.jsonl \\
        --failed data/processed/papers_failed.jsonl \\
        --api-url http://localhost:8080 \\
        --api-key $EXTRACTOR_API_KEY \\
        --batch-size 20 \\
        --dry-run   # process first 5 records only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from extractor.client import ExtractorClient


async def ingest(
    input_path: Path,
    output_path: Path,
    failed_path: Path,
    api_url: str,
    api_key: str,
    batch_size: int,
    dry_run: bool,
) -> None:
    records = []
    with open(input_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ✗ line {i+1}: JSON parse error — {e}", file=sys.stderr)

    if dry_run:
        records = records[:5]
        print(f"[dry-run] Processing first {len(records)} records.")

    print(f"Loaded {len(records)} records from {input_path}")

    # Validate required field
    valid = []
    for i, r in enumerate(records):
        if "section_text" not in r:
            print(f"  ✗ record {i}: missing 'section_text' — skipped")
        else:
            valid.append(r)
    print(f"Valid records: {len(valid)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    total_ok = 0
    total_failed = 0
    t0 = time.perf_counter()

    async with ExtractorClient(api_url, api_key=api_key) as client:
        if not await client.health():
            print(f"✗ Cannot reach extractor API at {api_url}. Start it first.", file=sys.stderr)
            sys.exit(1)

        with open(output_path, "w") as out_f, open(failed_path, "w") as fail_f:
            for batch_start in range(0, len(valid), batch_size):
                batch = valid[batch_start : batch_start + batch_size]
                sections = [r["section_text"] for r in batch]

                batch_resp = await client.extract_batch(sections)

                for record, result in zip(batch, batch_resp.results):
                    enriched = {
                        **record,
                        "extraction": {
                            "authors": result.authors,
                            "methodology": result.methodology,
                            "datasets_used": result.datasets_used,
                            "key_findings": result.key_findings,
                            "limitations": result.limitations,
                            "statistical_tests": result.statistical_tests,
                        },
                        "parse_error": result.parse_error,
                        "ingest_latency_s": result.latency_s,
                    }
                    if result.parse_error:
                        fail_f.write(json.dumps(enriched) + "\n")
                        total_failed += 1
                    else:
                        out_f.write(json.dumps(enriched) + "\n")
                        total_ok += 1

                pct = (batch_start + len(batch)) / len(valid) * 100
                print(
                    f"  {batch_start + len(batch)}/{len(valid)} ({pct:.0f}%) "
                    f"— {total_ok} ok, {total_failed} failed"
                )

    elapsed = time.perf_counter() - t0
    rate = len(valid) / elapsed if elapsed > 0 else 0

    print()
    print(f"Done in {elapsed:.1f}s ({rate:.1f} sections/sec)")
    print(f"  ✓  {total_ok} successful extractions → {output_path}")
    if total_failed:
        print(f"  ✗  {total_failed} failed extractions → {failed_path}")
    else:
        print("  ✓  No parse failures.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/papers_extracted.jsonl"),
        help="Output JSONL path (successful extractions)",
    )
    parser.add_argument(
        "--failed",
        type=Path,
        default=Path("data/processed/papers_failed.jsonl"),
        help="JSONL path for failed extractions",
    )
    parser.add_argument("--api-url", default="http://localhost:8080", help="Extractor API base URL")
    parser.add_argument("--api-key", default="", help="Bearer token (empty = auth disabled)")
    parser.add_argument("--batch-size", type=int, default=20, help="Sections per batch API call")
    parser.add_argument("--dry-run", action="store_true", help="Process only the first 5 records")
    args = parser.parse_args()

    asyncio.run(ingest(
        input_path=args.input,
        output_path=args.output,
        failed_path=args.failed,
        api_url=args.api_url,
        api_key=args.api_key,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
