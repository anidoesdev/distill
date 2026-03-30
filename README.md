# EXTRACTOR

Fine-tuned scientific paper information extractor. Beats GPT-4o-mini on structured
extraction accuracy at a fraction of the cost per call.

**Output schema:** `{authors, methodology, datasets_used, key_findings, limitations, statistical_tests}`

---

## Architecture

```
Scientific Paper Section (text)
          │
          ▼
┌─────────────────────┐
│   Distillation      │  Sessions 4–5
│   GPT-4 / Claude    │  ~2,000 (section, json) pairs
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Manual Audit      │  Session 6
│   Streamlit UI      │  ~200 human-verified examples
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   QLoRA SFT         │  Sessions 9–11
│   Qwen2.5 / Llama   │  4-bit NF4, TRL SFTTrainer
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   DPO Alignment     │  Sessions 15–17
│   TRL DPOTrainer    │  1,000 preference pairs
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Quantization      │  Sessions 19–20
│   AWQ / GGUF        │  Benchmark vs full precision
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   vLLM Server       │  Session 22
│   Continuous batch  │  Paged attention, OpenAI-compat API
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   FastAPI Wrapper   │  Sessions 23–24
│   /api/extract      │  Schema-constrained decoding, retry+repair
└─────────┬───────────┘
          │
          ▼
    Structured JSON
    {authors, methodology, datasets_used,
     key_findings, limitations, statistical_tests}
```

---

## Repository Layout

```
extractor/          installable Python package (training + serving share this)
  api/              FastAPI application
  model/            inference wrapper around vLLM
  schemas/          Pydantic output schema with strict validation
  utils/            structured logging, helpers
training/           training scripts (not shipped to prod)
  train_sft.py      QLoRA SFT with TRL SFTTrainer
  train_dpo.py      DPO alignment with TRL DPOTrainer
data/
  raw/              distilled pairs from teacher model (gitignored)
  processed/        tokenized HF Datasets (gitignored)
  eval/             golden evaluation set, 200 human-audited examples
scripts/
  run_baseline.py   zero-shot baseline evaluation
tests/              unit + integration + golden-output regression
```

---

## Quickstart

```bash
# Install for development
pip install -e ".[dev]"

# Run API server (no model loaded — stub)
uvicorn extractor.api.main:app --reload

# Check health
curl http://localhost:8080/health

# Run with Docker
docker compose up
```

---

## Session Log

| Session | Date       | Status | Deliverable                                      |
|---------|------------|--------|--------------------------------------------------|
| 1       | 2026-03-02 | ✓      | Project scaffold, Dockerfile, README             |
| 2       | 2026-03-03 | ✓      | Base model loaded, tokenizer, inference baseline |
| 3       | 2026-03-04 | ✓      | Output schema, extraction prompt, zero-shot eval |
| 4       | 2026-03-06 | ✓      | Teacher distillation — 2,000 examples            |
| 5       | 2026-03-07 | ✓      | Data validation pipeline                         |
| 6       | 2026-03-08 | ✓      | Manual audit Streamlit tool                      |
| 7       | 2026-03-10 | ✓      | Train/val/test split, data card                  |
| 8       | 2026-03-12 | ✓      | HF Datasets, tokenization, sequence analysis     |
| 9       | 2026-03-15 | ✓      | QLoRA SFT script + training config               |
| 10      | 2026-03-16 | ✓      | Smoke-test SFT run, loss curves                  |
| 11      | 2026-03-17 | ✓      | Full SFT run, W&B logging, checkpoint export     |
| 12      | 2026-03-20 | ✓      | Eval script, per-field EM + F1 metrics           |
| 13      | 2026-03-24 | ✓      | SFT vs base vs GPT-4o-mini comparison            |
| 14      | 2026-03-30 | ✓      | Buffer — polish, env checker, clean exports      |
| 15      | 2026-03-31 |        | DPO theory, Bradley-Terry, loss derivation       |
| 16      | 2026-04-01 |        | Preference dataset — 1,000 pairs                 |
| 17      | 2026-04-05 |        | DPO training run, reward margins                 |
| 18      | 2026-04-07 |        | DPO eval, alignment tax analysis                 |
| 19      | 2026-04-08 |        | Quantization theory, AWQ/GGUF                    |
| 20      | 2026-04-10 |        | Quantized vs full-precision benchmark            |
| 21      | 2026-04-11 |        | Model card                                       |
| 22      | 2026-04-16 |        | vLLM serving, continuous batching                |
| 23      | 2026-04-18 |        | FastAPI wrapper, auth, logging, retry+repair     |
| 24      | 2026-04-20 |        | Schema-constrained decoding (outlines)           |
| 25      | 2026-04-21 |        | Integration into scientific-rag-assistant        |
| 26      | 2026-04-23 |        | Frontend — "Extract Structure" button            |
| 27      | 2026-04-25 |        | Tests — unit, integration, regression            |
| 28      | 2026-04-26 |        | Cost + latency benchmark report                  |
| 29      | 2026-04-30 |        | Production polish — health, Prometheus, runbook  |
| 30      | 2026-05-01 |        | Mock interview — defend all decisions            |

---

## Model Candidates

| Model                  | Params | VRAM (4-bit) | Notes                              |
|------------------------|--------|--------------|------------------------------------|
| Qwen2.5-1.5B-Instruct  | 1.5B   | ~2 GB        | Fastest iteration, fits on Colab   |
| Qwen2.5-3B-Instruct    | 3B     | ~3 GB        | Better reasoning, still fast       |
| Llama-3.2-3B-Instruct  | 3B     | ~3 GB        | Strong instruction following       |
| Phi-3.5-mini-instruct  | 3.8B   | ~4 GB        | Highest quality, slowest           |

Final choice made in Session 2 based on zero-shot baseline results.
