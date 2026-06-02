"""Fetch paper abstracts from arXiv to use as distillation inputs.

arXiv Atom API docs: https://arxiv.org/help/api/user-manual
No API key required. Soft rate limit: ~3 requests/second.

Each record saved to data/raw/papers.jsonl has:
    {"arxiv_id", "title", "authors", "abstract", "categories", "published"}

The abstract is used as the section_text in distillation (session 4-5).
For a production dataset you'd also extract methods/conclusion sections from
PDFs, but abstracts are sufficient for the ~2,000 example target.

Usage:
    python scripts/fetch_papers.py                        # default mix
    python scripts/fetch_papers.py --n 500 --cat cs.LG   # single category
    python scripts/fetch_papers.py --n 2000 --resume     # resume after crash
"""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
OUT_PATH = Path("data/raw/papers.jsonl")

# Category mix for domain diversity — keeps the model from overfitting to one field
DEFAULT_CATEGORIES = [
    "cs.LG",    # machine learning
    "cs.CL",    # computation and language / NLP
    "stat.ML",  # statistics / ML
    "cs.CV",    # computer vision
    "q-bio.QM", # quantitative biology
    "physics.data-an",  # data analysis in physics
]


def fetch_page(category: str, start: int, max_results: int = 100) -> list[dict]:
    """Fetch one page of results from the arXiv API."""
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    records = []

    for entry in root.findall("atom:entry", NS):
        arxiv_id_url = entry.findtext("atom:id", "", NS) or ""
        arxiv_id = arxiv_id_url.split("/abs/")[-1].strip()

        title_el = entry.find("atom:title", NS)
        title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""

        abstract_el = entry.find("atom:summary", NS)
        abstract = (abstract_el.text or "").strip().replace("\n", " ") if abstract_el is not None else ""

        authors = [
            (name_el.text or "").strip()
            for author in entry.findall("atom:author", NS)
            for name_el in [author.find("atom:name", NS)]
            if name_el is not None and name_el.text
        ]

        cats = [
            t.get("term", "")
            for t in entry.findall("atom:category", NS)
        ]

        published_el = entry.find("atom:published", NS)
        published = (published_el.text or "")[:10] if published_el is not None else ""

        if not arxiv_id or not abstract or len(abstract) < 100:
            continue

        records.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "categories": cats,
                "published": published,
            }
        )

    return records


def load_existing_ids(path: Path) -> set[str]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="Target total records")
    parser.add_argument(
        "--cat",
        default=None,
        help="Single arXiv category. Omit to use default category mix.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip papers already in output file.",
    )
    args = parser.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = load_existing_ids(OUT_PATH) if args.resume else set()

    categories = [args.cat] if args.cat else DEFAULT_CATEGORIES
    per_cat = max(1, args.n // len(categories))

    total_written = len(existing_ids)
    print(f"Target: {args.n} records | Existing: {total_written} | Categories: {categories}")

    with OUT_PATH.open("a", encoding="utf-8") as fout:
        for cat in categories:
            if total_written >= args.n:
                break

            start = 0
            cat_written = 0
            cat_target = per_cat

            print(f"\n[{cat}] fetching up to {cat_target} records...")
            backoff = 15

            while cat_written < cat_target and total_written < args.n:
                try:
                    records = fetch_page(cat, start=start, max_results=50)
                except requests.RequestException as e:
                    wait = min(backoff, 120)
                    print(f"  Request failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                    backoff = min(backoff * 2, 120)
                    continue
                backoff = 15

                if not records:
                    print(f"  No more results at start={start}")
                    break

                new = 0
                for rec in records:
                    if rec["arxiv_id"] in existing_ids:
                        continue
                    existing_ids.add(rec["arxiv_id"])
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    cat_written += 1
                    total_written += 1
                    new += 1

                print(
                    f"  page start={start}: {len(records)} fetched, {new} new "
                    f"| total {total_written}/{args.n}"
                )
                start += 100

                # arXiv asks for ~3s between requests — be a good citizen
                time.sleep(3)

    print(f"\nDone. {total_written} records saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
