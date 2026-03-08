"""Manual audit tool for the distilled training dataset.

Run: streamlit run scripts/audit_app.py

Loads 200 examples from data/processed/train_clean.jsonl, lets you review
and correct each one, and saves the results to data/eval/human_audited.jsonl.

These 200 examples become the permanent eval set — they are removed from
training data. Every benchmark comparison in sessions 12–28 runs against this set.

State is checkpointed to data/eval/audit_decisions.json after every decision,
so you can close the browser and resume without losing work.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
CLEAN_PATH = Path("data/processed/train_clean.jsonl")
SAMPLE_PATH = Path("data/eval/audit_sample.json")      # the 200 selected examples
DECISIONS_PATH = Path("data/eval/audit_decisions.json") # approved/rejected/edited
AUDITED_PATH = Path("data/eval/human_audited.jsonl")    # final export

AUDIT_N = 200
FIELDS = ["authors", "datasets_used", "key_findings", "limitations", "statistical_tests"]
LIST_FIELDS = set(FIELDS)
STR_FIELDS = {"methodology"}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_clean_examples() -> list[dict]:
    if not CLEAN_PATH.exists():
        return []
    examples = []
    for line in CLEAN_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return examples


def get_or_create_sample() -> list[dict]:
    """Return the fixed 200-example audit sample, creating it if first run."""
    if SAMPLE_PATH.exists():
        return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))

    examples = load_clean_examples()
    if not examples:
        return []

    n = min(AUDIT_N, len(examples))
    sample = random.Random(42).sample(examples, n)  # fixed seed for reproducibility

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    return sample


def load_decisions() -> dict[str, dict]:
    if not DECISIONS_PATH.exists():
        return {}
    try:
        return json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_decisions(decisions: dict[str, dict]) -> None:
    DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DECISIONS_PATH.write_text(json.dumps(decisions, indent=2), encoding="utf-8")


def export_audited(sample: list[dict], decisions: dict[str, dict]) -> int:
    """Write approved/edited examples to the eval JSONL. Returns count written."""
    AUDITED_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with AUDITED_PATH.open("w", encoding="utf-8") as f:
        for ex in sample:
            aid = ex["arxiv_id"]
            dec = decisions.get(aid)
            if dec and dec["status"] in ("approved", "edited"):
                record = {
                    "arxiv_id": aid,
                    "title": ex.get("title", ""),
                    "section_text": ex.get("section_text", ""),
                    "extraction": dec["extraction"],
                    "audit_status": dec["status"],
                }
                f.write(json.dumps(record) + "\n")
                written += 1
    return written


# ── Streamlit app ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EXTRACTOR Audit",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("EXTRACTOR — Manual Audit Tool")
st.caption(
    "Review and correct teacher-generated extractions. "
    "These 200 examples become the permanent eval set."
)

# ── Session state init ────────────────────────────────────────────────────────
if "sample" not in st.session_state:
    st.session_state.sample = get_or_create_sample()

if "decisions" not in st.session_state:
    st.session_state.decisions = load_decisions()

if "idx" not in st.session_state:
    # Start at first unreviewed example
    reviewed = set(st.session_state.decisions.keys())
    unreviewed = [
        i for i, ex in enumerate(st.session_state.sample)
        if ex["arxiv_id"] not in reviewed
    ]
    st.session_state.idx = unreviewed[0] if unreviewed else 0

sample: list[dict] = st.session_state.sample
decisions: dict[str, dict] = st.session_state.decisions

if not sample:
    st.error(
        f"No examples found. Run `python scripts/validate_dataset.py` first "
        f"to generate {CLEAN_PATH}."
    )
    st.stop()

# ── Progress bar ──────────────────────────────────────────────────────────────
n_reviewed = sum(1 for ex in sample if ex["arxiv_id"] in decisions)
n_approved = sum(
    1 for d in decisions.values() if d["status"] in ("approved", "edited")
)
n_rejected = sum(1 for d in decisions.values() if d["status"] == "rejected")

progress_col, stats_col = st.columns([3, 1])
with progress_col:
    st.progress(n_reviewed / len(sample), text=f"{n_reviewed}/{len(sample)} reviewed")
with stats_col:
    st.metric("Approved", n_approved)

st.markdown("---")

# ── Navigation ────────────────────────────────────────────────────────────────
idx: int = st.session_state.idx
idx = max(0, min(idx, len(sample) - 1))

nav_cols = st.columns([1, 6, 1])
with nav_cols[0]:
    if st.button("← Prev", disabled=idx == 0):
        st.session_state.idx = idx - 1
        st.rerun()
with nav_cols[1]:
    st.markdown(
        f"<div style='text-align:center; font-size:0.9em; color:gray;'>"
        f"Example {idx + 1} of {len(sample)}</div>",
        unsafe_allow_html=True,
    )
with nav_cols[2]:
    if st.button("Next →", disabled=idx == len(sample) - 1):
        st.session_state.idx = idx + 1
        st.rerun()

# ── Current example ───────────────────────────────────────────────────────────
example = sample[idx]
arxiv_id: str = example["arxiv_id"]
existing_decision = decisions.get(arxiv_id)
existing_status = existing_decision["status"] if existing_decision else None

# Status badge
if existing_status == "approved":
    st.success(f"✓ Approved  |  arXiv: {arxiv_id}")
elif existing_status == "edited":
    st.info(f"✎ Edited  |  arXiv: {arxiv_id}")
elif existing_status == "rejected":
    st.error(f"✗ Rejected  |  arXiv: {arxiv_id}")
else:
    st.warning(f"Pending  |  arXiv: {arxiv_id}")

st.markdown(f"**{example.get('title', '(no title)')}**")

# ── Two-column layout: paper text | extraction ────────────────────────────────
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Paper Section")
    st.text_area(
        "section_text",
        value=example.get("section_text", ""),
        height=400,
        disabled=True,
        label_visibility="collapsed",
    )

with right_col:
    st.subheader("Extracted Fields (editable)")

    # Load either the existing edited extraction or the original
    if existing_decision and existing_decision["status"] in ("approved", "edited"):
        base_ext = existing_decision["extraction"]
    else:
        base_ext = example.get("extraction", {})

    # Authors (comma-separated for easy editing)
    authors_raw = st.text_input(
        "Authors (comma-separated)",
        value=", ".join(base_ext.get("authors", [])),
        key=f"authors_{idx}",
    )

    methodology_val = st.text_area(
        "Methodology",
        value=base_ext.get("methodology", ""),
        height=100,
        key=f"methodology_{idx}",
    )

    datasets_raw = st.text_input(
        "Datasets used (comma-separated)",
        value=", ".join(base_ext.get("datasets_used", [])),
        key=f"datasets_{idx}",
    )

    findings_raw = st.text_area(
        "Key findings (one per line)",
        value="\n".join(base_ext.get("key_findings", [])),
        height=100,
        key=f"findings_{idx}",
    )

    limitations_raw = st.text_area(
        "Limitations (one per line)",
        value="\n".join(base_ext.get("limitations", [])),
        height=80,
        key=f"limitations_{idx}",
    )

    stats_raw = st.text_input(
        "Statistical tests (comma-separated)",
        value=", ".join(base_ext.get("statistical_tests", [])),
        key=f"stats_{idx}",
    )

# ── Collect current field values ──────────────────────────────────────────────
def split_commas(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def split_lines(s: str) -> list[str]:
    return [x.strip() for x in s.splitlines() if x.strip()]

current_extraction = {
    "authors": split_commas(authors_raw),
    "methodology": methodology_val.strip(),
    "datasets_used": split_commas(datasets_raw),
    "key_findings": split_lines(findings_raw),
    "limitations": split_lines(limitations_raw),
    "statistical_tests": split_commas(stats_raw),
}

# Check if user edited any field compared to original
original_ext = example.get("extraction", {})
is_edited = current_extraction != {
    "authors": original_ext.get("authors", []),
    "methodology": original_ext.get("methodology", ""),
    "datasets_used": original_ext.get("datasets_used", []),
    "key_findings": original_ext.get("key_findings", []),
    "limitations": original_ext.get("limitations", []),
    "statistical_tests": original_ext.get("statistical_tests", []),
}

# ── Decision buttons ──────────────────────────────────────────────────────────
st.markdown("---")
btn_cols = st.columns([2, 2, 2, 3])

approve_label = "✎ Save Edits + Approve" if is_edited else "✓ Approve"
with btn_cols[0]:
    if st.button(approve_label, type="primary", use_container_width=True):
        status = "edited" if is_edited else "approved"
        decisions[arxiv_id] = {"status": status, "extraction": current_extraction}
        save_decisions(decisions)
        st.session_state.decisions = decisions
        # Auto-advance to next unreviewed
        next_unreviewed = next(
            (i for i in range(idx + 1, len(sample))
             if sample[i]["arxiv_id"] not in decisions),
            idx + 1 if idx + 1 < len(sample) else idx,
        )
        st.session_state.idx = next_unreviewed
        st.rerun()

with btn_cols[1]:
    if st.button("✗ Reject", use_container_width=True):
        decisions[arxiv_id] = {"status": "rejected", "extraction": {}}
        save_decisions(decisions)
        st.session_state.decisions = decisions
        next_idx = idx + 1 if idx + 1 < len(sample) else idx
        st.session_state.idx = next_idx
        st.rerun()

with btn_cols[2]:
    if st.button("Skip (decide later)", use_container_width=True):
        st.session_state.idx = idx + 1 if idx + 1 < len(sample) else idx
        st.rerun()

with btn_cols[3]:
    if st.button("Export Approved Set", use_container_width=True):
        count = export_audited(sample, decisions)
        st.success(
            f"Exported {count} examples to {AUDITED_PATH}. "
            "These are your eval set for sessions 12–28."
        )

# ── Sidebar: jump to example by index ────────────────────────────────────────
with st.sidebar:
    st.header("Navigation")
    jump = st.number_input(
        "Jump to example #", min_value=1, max_value=len(sample), value=idx + 1
    )
    if st.button("Go"):
        st.session_state.idx = jump - 1
        st.rerun()

    st.markdown("---")
    st.header("Session Summary")
    st.write(f"Total: {len(sample)}")
    st.write(f"Reviewed: {n_reviewed}")
    st.write(f"Approved: {n_approved}")
    st.write(f"Rejected: {n_rejected}")
    st.write(f"Remaining: {len(sample) - n_reviewed}")

    st.markdown("---")
    st.caption(f"Sample locked: {SAMPLE_PATH.exists()}")
    st.caption(f"Checkpoint: {DECISIONS_PATH}")
    st.caption(f"Output: {AUDITED_PATH}")
