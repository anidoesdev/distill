"""FastAPI application entry point.

Run locally: uvicorn extractor.api.main:app --host 0.0.0.0 --port 8080 --reload
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from extractor.config import settings
from extractor.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging(settings.log_level)
    logger.info("extractor api starting", extra={"model": settings.model_name})
    yield
    logger.info("extractor api shutting down")


app = FastAPI(
    title="EXTRACTOR",
    description="Fine-tuned scientific paper information extractor",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# /api/extract endpoint added in session 23 (after model is trained and served)
