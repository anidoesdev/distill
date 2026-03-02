# EXTRACTOR serving image
# Build: docker build -t extractor:latest .
# Run:   docker compose up

# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
# Install only the serving subset of requirements (not torch/training deps)
COPY requirements.txt .
RUN pip install \
    fastapi==0.115.6 \
    uvicorn[standard]==0.32.1 \
    pydantic==2.10.3 \
    pydantic-settings==2.7.0 \
    httpx==0.28.1 \
    tenacity==9.0.0 \
    orjson==3.10.12 \
    python-dotenv==1.0.1

# ── Application ───────────────────────────────────────────────────────────────
COPY extractor/ ./extractor/
COPY pyproject.toml .
RUN pip install -e . --no-deps

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8080

# Non-root user for prod
RUN adduser --disabled-password --gecos "" appuser
USER appuser

CMD ["uvicorn", "extractor.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
