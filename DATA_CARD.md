# EXTRACTOR Dataset Card

Dataset for supervised fine-tuning (SFT) of a scientific paper information extractor.
Each example is a (paper_section, structured_extraction) pair.

---

## Summary

| Property         | Value                                          |
|------------------|------------------------------------------------|
| Task             | Structured extraction from scientific text     |
| Output schema    | 6 fields: authors, methodology, datasets_used, key_findings, limitations, statistical_tests |
| Total examples   | ~2,000 (after validation)                      |
| Teacher model    | GPT-4o-mini (OpenAI)                           |
| Source text      | arXiv abstracts (2024–2026)                    |
| Languages        | English                                        |
| License          | arXiv non-exclusive license (source text); generated labels are project property |

---

## Source

Paper sections are abstracts fetched from the arXiv Atom API. No authentication
is required. arXiv allows bulk access for non-commercial research use.

**Categories fetched:**
- `cs.LG` — machine learning
- `cs.CL` — computation and language / NLP
- `stat.ML` — statistical machine learning
- `cs.CV` — computer vision
- `q-bio.QM` — quantitative biology
- `physics.data-an` — data analysis in physics

Category diversity is intentional: a model trained only on CS papers will fail
on biology or physics papers. The diversity forces the model to learn general
extraction behavior rather than domain-specific templates.

---

## Collection and Generation

1. **Paper fetch** (`scripts/fetch_papers.py`): arXiv API, sorted by submission date
   descending. Deduplicated by `arxiv_id`. Rate-limited to 3 req/s.

2. **Teacher labeling** (`scripts/generate_dataset.py`): GPT-4o-mini called with
   the frozen system prompt from `extractor/prompt.py` (`EXTRACTION_SYSTEM_PROMPT`).
   Parameters: `temperature=0.0`, `max_tokens=1024`. Cost: ~$0.50 for 2,000 examples.

3. **Automated validation** (`scripts/validate_dataset.py`): schema validity,
   semantic checks (author quality, methodology length, findings presence), content
   injection detection, author cross-reference against arXiv metadata.

4. **Human audit** (`scripts/audit_app.py`): 200 examples reviewed and corrected
   by one annotator. These 200 examples are the permanent eval set and are never
   used for training.

5. **Split** (`scripts/create_splits.py`): remaining examples split 90/10 train/val
   with seed=42. See `data/processed/split_manifest.json` for exact arxiv_id
   assignment.

---

## Splits

| Split | Size | Source | Purpose |
|-------|------|--------|---------|
| train | ~1,440 | auto-validated, teacher-labeled | SFT training (sessions 9–11) |
| val | ~160 | auto-validated, teacher-labeled | Early stopping, hyperparameter tuning |
| eval | 200 | human-audited, human-corrected | All benchmark comparisons (sessions 12–28) |

**Reproducibility:** `data/processed/split_manifest.json` records the exact
`arxiv_id` assigned to each split. Committed to git. Any rebuild of the data files
can be verified against this manifest.

**Leakage prevention:** The 200 eval `arxiv_id`s are explicitly removed from the
training pool before splitting. `extractor.data.splits.verify_no_leakage()` asserts
zero overlap at the start of each training script.

---

## Schema

```json
{
  "authors": ["list of author full names"],
  "methodology": "one to three sentence description of methods",
  "datasets_used": ["list of dataset names"],
  "key_findings": ["list of main results"],
  "limitations": ["list of stated limitations"],
  "statistical_tests": ["list of statistical tests mentioned"]
}
```

Fields not mentioned in the source text are set to empty list or empty string.
The schema is versioned in `extractor/schemas/extraction.py`. Any change to field
names or types requires regenerating the training data.

---

## Field Presence Rates (approximate, varies by run)

These rates reflect what's in paper abstracts. Limitations and statistical tests
are less commonly stated in abstracts than in methods/results sections. A future
dataset version using full-text sections would have higher presence rates.

| Field              | Train presence | Eval presence |
|--------------------|---------------|---------------|
| authors            | ~85%          | ~95% (human-corrected) |
| methodology        | ~90%          | ~95% |
| datasets_used      | ~75%          | ~85% |
| key_findings       | ~92%          | ~98% |
| limitations        | ~55%          | ~75% |
| statistical_tests  | ~45%          | ~65% |

---

## Known Limitations

1. **Abstract-only.** The source text is abstracts, not full sections. Abstracts
   often omit detailed methodology and rarely state limitations explicitly. The
   model trained on this data will extrapolate these behaviors to full-section
   inputs, but may be less reliable on detailed methods sections.

2. **Teacher hallucination.** GPT-4o-mini sometimes invents plausible-sounding
   author names or datasets not present in the abstract. The automated cross-
   reference validator catches ~70% of these; human audit catches the rest
   for the 200 eval examples. Training examples may contain uncaught hallucinations.

3. **English-only.** arXiv is predominantly English. The model has no multilingual
   extraction capability.

4. **Recent papers only.** Papers from 2024–2026. May not generalize to older
   papers with different citation norms.

5. **CS/ML-heavy.** Even with category diversity, the dataset skews toward
   machine learning papers because that's the dominant category on arXiv. Biology
   and physics extraction will be less accurate.

---

## Usage

```python
from extractor.data.splits import load_split, verify_no_leakage

verify_no_leakage()  # call at start of any training script

train = load_split("train")
val = load_split("val")
eval_set = load_split("eval")

# Each example:
# {
#   "arxiv_id": "2310.xxxxx",
#   "title": "...",
#   "section_text": "...",
#   "extraction": {"authors": [...], "methodology": "...", ...},
#   "metadata": {"teacher_model": "gpt-4o-mini", ...}  # train/val only
# }
```

---

## Citation

This dataset was generated for the EXTRACTOR portfolio project. If you use it,
reference this repository and the arXiv API terms of use.
