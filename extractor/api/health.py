"""Health check endpoints for Kubernetes liveness and readiness probes.

Endpoints:
  GET /health/live    — liveness probe: always 200 while process is up
  GET /health/ready   — readiness probe: 200 if vLLM is reachable, 503 if not
  GET /health         — backwards-compatible alias for /health/live

The old GET /health on main.py is kept for backwards compatibility with
existing docker-compose healthchecks. The /health/live and /health/ready
paths are the canonical Kubernetes probe targets.

Kubernetes pod spec example:
  livenessProbe:
    httpGet:
      path: /health/live
      port: 8080
    initialDelaySeconds: 5
    periodSeconds: 10

  readinessProbe:
    httpGet:
      path: /health/ready
      port: 8080
    initialDelaySeconds: 10
    periodSeconds: 15
    failureThreshold: 3
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Response

from extractor.config import settings
from extractor.model.vllm_client import VLLMClient
from extractor.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])

_start_time = time.time()


@router.get("/health/live")
async def liveness() -> dict:
    """Liveness probe — returns 200 while the process is running.

    Never queries external dependencies. k8s uses this to decide whether
    to restart the container. Only fails if the process itself is hung.
    """
    return {
        "status": "live",
        "uptime_s": round(time.time() - _start_time, 1),
    }


@router.get("/health/ready")
async def readiness(response: Response) -> dict:
    """Readiness probe — returns 200 only if the service can handle traffic.

    Checks:
      1. vLLM server is reachable and responding to /health
      2. (future) GPU memory is not critically low

    Returns 503 with details if any check fails. k8s will stop routing
    traffic to this pod until readiness is restored.
    """
    checks: dict[str, bool] = {}

    # Check 1: vLLM connectivity
    try:
        async with VLLMClient() as client:
            vllm_ok = await client.health()
        checks["vllm_reachable"] = vllm_ok
    except Exception as exc:
        logger.warning("readiness check: vLLM error", extra={"error": str(exc)})
        checks["vllm_reachable"] = False

    all_ok = all(checks.values())
    if not all_ok:
        response.status_code = 503
        logger.warning("readiness probe failed", extra={"checks": checks})

    return {
        "status": "ready" if all_ok else "not_ready",
        "checks": checks,
        "model": settings.model_name,
        "uptime_s": round(time.time() - _start_time, 1),
    }
