"""Programmatic degradation of extraction results to create rejected preference pairs.

Each function takes a valid ExtractionResult dict and returns a degraded version
that is measurably worse but still looks plausible. The degraded output is used
as the 'rejected' side of a preference pair.

Design constraints:
  - Rejected must be valid JSON (same schema as chosen) — invalid JSON is too easy
    for the model to distinguish and produces noisy gradients
  - Degradation must lower the F1 score relative to the human reference
  - No single strategy should dominate the dataset (variety = better generalization)
"""

from __future__ import annotations

import random
import re


def drop_authors(extraction: dict) -> dict:
    """Remove all but the first author, or clear the list entirely."""
    d = extraction.copy()
    authors = d.get("authors", [])
    if len(authors) > 1:
        d["authors"] = [authors[0]]
    else:
        d["authors"] = []
    return d


def truncate_findings(extraction: dict) -> dict:
    """Keep only the first key finding, drop the rest."""
    d = extraction.copy()
    findings = d.get("key_findings", [])
    d["key_findings"] = findings[:1] if findings else []
    return d


def clear_field(extraction: dict, field: str) -> dict:
    """Wipe a field to its empty value."""
    d = extraction.copy()
    if isinstance(d.get(field), list):
        d[field] = []
    elif isinstance(d.get(field), str):
        d[field] = ""
    return d


def truncate_methodology(extraction: dict, keep_fraction: float = 0.4) -> dict:
    """Cut methodology to first keep_fraction of words."""
    d = extraction.copy()
    text = d.get("methodology", "")
    if not text:
        return d
    words = text.split()
    cutoff = max(1, int(len(words) * keep_fraction))
    d["methodology"] = " ".join(words[:cutoff]) + "..."
    return d


def shuffle_datasets(extraction: dict) -> dict:
    """Randomly reorder datasets_used items and corrupt one name."""
    d = extraction.copy()
    datasets = list(d.get("datasets_used", []))
    if not datasets:
        return d
    random.shuffle(datasets)
    if datasets:
        datasets[0] = datasets[0].lower().replace(" ", "_")
    d["datasets_used"] = datasets
    return d


def add_generic_noise(extraction: dict) -> dict:
    """Replace one findings item with a vague generic sentence."""
    d = extraction.copy()
    findings = list(d.get("key_findings", []))
    generic = [
        "Results show improved performance.",
        "The model achieved good accuracy.",
        "Further analysis is needed.",
        "Results were consistent with expectations.",
    ]
    if findings:
        idx = random.randrange(len(findings))
        findings[idx] = random.choice(generic)
    d["key_findings"] = findings
    return d


def drop_limitations(extraction: dict) -> dict:
    """Clear the limitations field (common model failure — often left empty)."""
    d = extraction.copy()
    d["limitations"] = []
    return d


def drop_statistical_tests(extraction: dict) -> dict:
    """Clear statistical_tests (another commonly dropped field)."""
    d = extraction.copy()
    d["statistical_tests"] = []
    return d


# ── Strategy registry ─────────────────────────────────────────────────────────

# Each strategy is (name, function, weight) where weight controls sampling
# probability. Heavier weight = more examples with this degradation type.
STRATEGIES: list[tuple[str, callable, float]] = [
    ("drop_authors",           drop_authors,           1.5),
    ("truncate_findings",      truncate_findings,      1.5),
    ("clear_methodology",      lambda e: clear_field(e, "methodology"),    1.0),
    ("clear_datasets",         lambda e: clear_field(e, "datasets_used"),  0.8),
    ("truncate_methodology",   truncate_methodology,   1.2),
    ("add_generic_noise",      add_generic_noise,      1.0),
    ("drop_limitations",       drop_limitations,       0.7),
    ("drop_statistical_tests", drop_statistical_tests, 0.7),
]

_names    = [s[0] for s in STRATEGIES]
_fns      = [s[1] for s in STRATEGIES]
_weights  = [s[2] for s in STRATEGIES]


def degrade(extraction: dict, rng: random.Random | None = None) -> tuple[dict, str]:
    """Apply one randomly selected degradation strategy.

    Returns (degraded_extraction, strategy_name).
    """
    if rng is None:
        rng = random.Random()
    fn, name = rng.choices(list(zip(_fns, _names)), weights=_weights, k=1)[0]
    return fn(extraction), name


def degrade_composite(extraction: dict, rng: random.Random | None = None, n: int = 2) -> tuple[dict, list[str]]:
    """Apply n randomly selected, non-duplicate degradation strategies.

    Returns (degraded_extraction, [strategy_names]).
    Composite degradation creates harder negatives with lower F1 scores.
    """
    if rng is None:
        rng = random.Random()
    chosen_indices = rng.sample(range(len(STRATEGIES)), k=min(n, len(STRATEGIES)))
    strategies_applied = []
    result = extraction.copy()
    for i in chosen_indices:
        result = _fns[i](result)
        strategies_applied.append(_names[i])
    return result, strategies_applied
