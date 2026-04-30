# EXTRACTOR — Production Runbook

**Service:** EXTRACTOR API  
**Owner:** ML Platform  
**On-call rotation:** `#ml-oncall` Slack channel  
**Repo:** `d:/distill`  
**Last updated:** 2026-04-30

---

## Service Overview

EXTRACTOR is a FastAPI service that wraps a fine-tuned Qwen2.5-1.5B-Instruct model
(quantized to AWQ INT4) served by vLLM. It accepts scientific paper section text and
returns structured JSON (authors, methodology, datasets, findings, limitations, tests).

**Production stack:**
```
Client → extractor-api (port 8080) → vLLM (port 8000, AWQ model)
                                    └→ /demo Gradio UI
                                    └→ /metrics Prometheus
```

---

## Starting and Stopping

### Start everything
```bash
docker compose up -d
docker compose logs -f extractor-api   # tail logs
```

### Start only vLLM (for debugging API separately)
```bash
docker compose up -d vllm
curl http://localhost:8000/health       # expect {"status":"ok"}
```

### Start only extractor-api (vLLM must already be running)
```bash
docker compose up -d extractor-api
curl http://localhost:8080/health/live  # expect {"status":"live",...}
curl http://localhost:8080/health/ready # expect {"status":"ready",...}
```

### Stop everything
```bash
docker compose down
```

### Restart a single service
```bash
docker compose restart extractor-api   # reload without touching vLLM
```

### Check environment before starting
```bash
python scripts/check_env.py --phase vllm
```

---

## Health Checks

| Endpoint | Purpose | Expected |
|----------|---------|---------|
| `GET /health/live` | Liveness (process up?) | `{"status":"live"}` 200 |
| `GET /health/ready` | Readiness (vLLM reachable?) | `{"status":"ready"}` 200 |
| `GET /metrics` | Prometheus scrape target | text/plain |

### Quick health check
```bash
# All green:
curl -s http://localhost:8080/health/ready | python -m json.tool

# If vLLM is down, expect:
# {"status": "not_ready", "checks": {"vllm_reachable": false}, ...}  HTTP 503
```

---

## Common Failure Modes

### 1. vLLM not reachable (most common)

**Symptom:** `/health/ready` returns 503; `/api/extract` returns 500.

**Diagnose:**
```bash
docker compose ps vllm              # is it running?
docker compose logs vllm --tail=50  # look for OOM or model load errors
curl http://localhost:8000/health    # direct health check
```

**Fixes:**
```bash
# If OOM (GPU out of memory):
docker compose restart vllm         # frees GPU memory and reloads
# If checkpoint missing:
ls checkpoints/awq/                 # must contain config.json, model.safetensors
python scripts/quantize_awq.py      # regenerate if missing

# If stuck loading (>120s):
docker compose down vllm && docker compose up -d vllm
```

**Alert threshold:** readiness check fails for >60s → page oncall.

---

### 2. Parse failure rate spike

**Symptom:** `extractor_parse_failures_total` counter growing faster than normal.
Normal rate: <2% of requests. Alert at >10% over 5 minutes.

**PromQL alert:**
```promql
rate(extractor_parse_failures_total[5m])
  / rate(extractor_requests_total{endpoint="/api/extract"}[5m])
  > 0.10
```

**Diagnose:**
```bash
# Check recent model outputs
docker compose logs extractor-api --tail=100 | grep "parse failed"
# Check repair success rate
docker compose logs extractor-api --tail=100 | grep "repair"
```

**Root causes and fixes:**
- Model producing truncated output → increase `MAX_NEW_TOKENS` env var
- vLLM returning finish_reason=length → check `max-model-len` in compose file
- Prompt drift (someone changed `extractor/prompt.py`) → revert and redeploy

---

### 3. High latency (p99 > 5s)

**Symptom:** `histogram_quantile(0.99, rate(extractor_request_latency_seconds_bucket[5m]))` > 5.

**Diagnose:**
```bash
# Check GPU utilization
docker compose exec vllm nvidia-smi
# Check vLLM queue
curl http://localhost:8000/metrics | grep num_requests_waiting
```

**Fixes:**
- High queue depth: add concurrency via `--max-num-seqs` in vLLM command
- GPU memory pressure: reduce `--gpu-memory-utilization` from 0.85 to 0.80
- Swap CPU pressure: check system memory with `free -h`

**Alert threshold:** p99 > 5s for >5 minutes.

---

### 4. Container keeps restarting

**Symptom:** `docker compose ps` shows `Restarting` status.

**Diagnose:**
```bash
docker compose logs extractor-api --tail=100
# Common causes: import error, port conflict, missing .env
```

**Fixes:**
```bash
# Port conflict:
lsof -i :8080   # find what's using the port
# Missing env:
cp .env.example .env && vim .env   # add required values
# Bad config:
python -c "from extractor.config import settings; print(settings)"
```

---

### 5. Auth errors (401) from legitimate callers

**Symptom:** Callers getting 401 despite having API keys.

**Diagnose:**
```bash
# Check if API key is set
docker compose exec extractor-api env | grep API_KEY
# Test with key
curl -H "Authorization: Bearer $EXTRACTOR_API_KEY" \
     http://localhost:8080/api/info
```

**Fix:** Ensure `API_KEY` env var matches caller's token. Rotating keys requires
updating both the server env var and all caller configurations.

---

## Key Metrics and Alert Thresholds

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| `rate(extractor_requests_total[1m])` | >0 | — | 0 for >5min |
| `p99 request_latency_seconds` | <2s | <5s | >5s |
| `parse_failure_rate` | <2% | <10% | >10% |
| `rate(repair_attempts_total[5m])` | <5/min | <20/min | >20/min |
| vLLM `num_requests_waiting` | 0 | <5 | >10 |

---

## Rollback Procedure

### Roll back extractor-api to previous image
```bash
docker compose down extractor-api
# Edit docker-compose.yml: change image tag to previous version
docker compose up -d extractor-api
```

### Roll back to SFT model (from DPO)
```bash
# In docker-compose.yml, change:
#   --model /checkpoints/awq
# to:
#   --model /checkpoints/sft_awq   (if available)
docker compose restart vllm
```

### Emergency: disable guided decoding
```bash
# Set USE_GUIDED_DECODING=0 in environment and restart
docker compose down extractor-api
USE_GUIDED_DECODING=0 docker compose up -d extractor-api
```

---

## Prometheus Scrape Configuration

Add to `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: extractor
    static_configs:
      - targets: ['extractor-api:8080']
    scrape_interval: 15s
    metrics_path: /metrics
```

**Key PromQL queries:**
```promql
# Request rate
rate(extractor_requests_total[1m])

# p99 latency
histogram_quantile(0.99,
  rate(extractor_request_latency_seconds_bucket{endpoint="/api/extract"}[5m]))

# Parse failure rate
rate(extractor_parse_failures_total[5m])
  / rate(extractor_requests_total{endpoint="/api/extract"}[5m])

# vLLM latency p95
histogram_quantile(0.95,
  rate(extractor_vllm_latency_seconds_bucket[5m]))
```

---

## Useful Commands Reference

```bash
# Check environment
python scripts/check_env.py --phase vllm

# Run eval suite against live model
python scripts/eval_sft.py --model-url http://localhost:8000

# Benchmark latency
python scripts/benchmark_latency.py --n 50 --concurrency 1

# Full cost report
python -m scripts.cost_report --reference-only

# Run all tests
python -m pytest tests/ -q

# Check quantization artifacts
ls -lh checkpoints/awq/
ls -lh checkpoints/dpo-merged/
```
