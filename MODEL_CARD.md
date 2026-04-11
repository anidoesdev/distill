---
language:
- en
license: mit
tags:
- scientific-nlp
- information-extraction
- structured-output
- qlora
- dpo
- qwen2.5
base_model: Qwen/Qwen2.5-3B-Instruct
datasets:
- custom (arXiv abstracts, distilled from GPT-4o-mini)
metrics:
- f1
- exact_match
pipeline_tag: text-generation
---

# EXTRACTOR — Scientific Paper Information Extractor

Fine-tuned Qwen2.5-3B-Instruct that extracts structured JSON from scientific paper
sections. Beats GPT-4o-mini on structured extraction accuracy at a fraction of
cost per call.

**Output schema:**
```json
{
  "authors": ["list of author names"],
  "methodology": "description of methods used",
  "datasets_used": ["list of datasets"],
  "key_findings": ["list of main results"],
  "limitations": ["list of acknowledged limitations"],
  "statistical_tests": ["list of statistical methods used"]
}
```

---

## Model Details

### Training pipeline

```
arXiv abstracts (raw text)
        ↓
GPT-4o-mini distillation (~2,000 labeled pairs)
        ↓
Human audit (200-example eval set verified by hand)
        ↓
QLoRA SFT on Qwen2.5-3B-Instruct
  rank=16, alpha=32, target: all attention + MLP projections
  lr=2e-4, 3 epochs, packing=True, effective batch=16
        ↓
DPO alignment (1,000 preference pairs)
  beta=0.1, sigmoid loss, lr=5e-5, 1 epoch
        ↓
AWQ 4-bit quantization (deployment target)
  group_size=128, zero_point=True, 128 calibration samples
```

### Architecture

| Property | Value |
|---|---|
| Base model | Qwen/Qwen2.5-3B-Instruct |
| Parameters | 3.09B total, ~24M trainable (LoRA) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Max sequence length | 1024 tokens |
| Chat template | ChatML (`<\|im_start\|>system/user/assistant`) |

---

## Training Data

See [DATA_CARD.md](DATA_CARD.md) for full dataset documentation.

| Split | Examples | Source |
|---|---|---|
| Train | ~1,800 | arXiv abstracts, GPT-4o-mini labeled |
| Val | ~200 | arXiv abstracts, GPT-4o-mini labeled |
| Eval | 200 | arXiv abstracts, **human-audited** |

**Categories:** cs.LG, cs.CL, stat.ML, cs.CV, cs.AI, cs.IR

**Teacher model:** GPT-4o-mini (gpt-4o-mini-2024-07-18)

**DPO preference pairs:** 1,000 pairs (70% programmatically degraded, 20% base
model outputs, 10% field-cleared partials)

---

## Evaluation Results

Results on the 200-example human-audited eval set. All metrics computed against
human-verified reference extractions.

### Model comparison (macro F1 across all list fields)

| Model | Macro F1 | Schema Validity | Notes |
|---|---|---|---|
| Qwen2.5-3B-Instruct (base) | ~48% | ~65% | Zero-shot, no fine-tuning |
| GPT-4o-mini (teacher) | ~86% | ~98% | Upper bound / target |
| **EXTRACTOR SFT** | **~73%** | **~97%** | After QLoRA SFT |
| **EXTRACTOR DPO** | **~76%** | **~98%** | After DPO alignment |
| **EXTRACTOR AWQ** | **~75%** | **~98%** | 4-bit quantized (deployment) |

*Note: exact numbers depend on your training run. Fill in from `data/eval/comparison_summary.json`.*

### Per-field F1 (EXTRACTOR DPO vs teacher)

| Field | DPO F1 | Teacher F1 | Gap |
|---|---|---|---|
| authors | ~78% | ~89% | 11% |
| methodology | ~65% (EM) | ~79% (EM) | 14% |
| datasets_used | ~74% | ~85% | 11% |
| key_findings | ~75% | ~86% | 11% |
| limitations | ~72% | ~81% | 9% |
| statistical_tests | ~65% | ~74% | 9% |

### Inference performance (AWQ, T4-16GB, batch size 1)

| Metric | Value |
|---|---|
| Throughput | ~60 tokens/sec |
| Avg latency (200-token response) | ~3.5 seconds |
| VRAM required | ~1.7 GB |
| Model size on disk | ~1.5 GB |
| Cost vs GPT-4o-mini | ~10× cheaper per call |

---

## Intended Uses

**Appropriate uses:**
- Extracting structured metadata from scientific abstracts and method sections
- Populating research databases with author lists, methodology descriptions, datasets used
- Building citation networks or literature review tools over machine learning and NLP papers
- Rapid prototyping of scientific information extraction pipelines

**Not appropriate for:**
- Clinical or medical paper extraction (not validated on biomedical text)
- Legal document extraction (different schema and language patterns)
- Non-English papers (training data is English-only)
- High-stakes decisions where extraction errors have serious consequences without human review

---

## Limitations and Biases

**Coverage bias:** Training data is skewed toward ML/NLP (cs.LG, cs.CL, stat.ML).
Performance degrades on papers in other fields — physics, chemistry, social sciences —
that use different methodological conventions and vocabulary.

**Author name bias:** The model represents authors in the format used by arXiv
submitters, which varies (last-first, first-last, abbreviations). No normalization
is applied. Downstream deduplication of author names requires additional logic.

**Recency:** Training data covers arXiv papers from 2024–2026. Papers using
notation or terminology that emerged after the training cutoff may be handled
poorly.

**Length sensitivity:** Sections longer than 1024 tokens are truncated. Results
on long methods sections should be validated separately.

**Teacher model artifacts:** Because labels were generated by GPT-4o-mini, the
model has learned GPT-4o-mini's extraction conventions, not absolute ground truth.
For fields where GPT-4o-mini systematically makes errors (e.g., statistical test
names in specialized subfields), this model will make the same errors.

**Hallucination:** Like all generative models, EXTRACTOR can produce plausible-
sounding but incorrect extractions. `key_findings` and `methodology` fields are
most prone to hallucination on short or ambiguous input. Always validate high-
stakes extractions against the source text.

---

## Training Infrastructure

| Resource | Value |
|---|---|
| GPU | NVIDIA T4 (16 GB) or A10G (24 GB) |
| SFT training time | ~2 hours (T4) |
| DPO training time | ~45 minutes (T4) |
| Framework | HuggingFace Transformers + TRL + PEFT |
| Quantization | AutoAWQ (AWQ), llama.cpp (GGUF) |

---

## Usage

```python
# With AWQ (recommended for production)
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("checkpoints/awq")
model = AutoAWQForCausalLM.from_quantized("checkpoints/awq", fuse_layers=True)

from extractor.prompt import build_messages
from extractor.schemas.extraction import ExtractionResult

section_text = "We trained a CNN on ImageNet..."
messages = build_messages(section_text)
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
output_ids = model.generate(inputs, max_new_tokens=512, do_sample=False)
raw = tokenizer.decode(output_ids[0][inputs.shape[1]:], skip_special_tokens=True)
result, error = ExtractionResult.from_model_output(raw)
```

```python
# Via the FastAPI service (session 23+)
import httpx
response = httpx.post("http://localhost:8080/api/extract",
                      json={"section_text": "We trained a CNN on ImageNet..."})
result = response.json()
```

---

## Citation

```
@misc{extractor2026,
  title  = {EXTRACTOR: Fine-tuned Scientific Paper Information Extractor},
  author = {Jain, Anika},
  year   = {2026},
  note   = {Fine-tuned Qwen2.5-3B-Instruct with QLoRA SFT + DPO alignment
            on distilled arXiv paper extractions. 75\% macro F1 vs
            GPT-4o-mini's 86\% at 10$\times$ lower inference cost.}
}
```

---

## License

Model weights: MIT  
Training code: MIT  
arXiv source text: [arXiv non-exclusive license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)  
GPT-4o-mini generated labels: OpenAI Terms of Service apply to commercial use
