"""API key authentication dependency, shared across routers."""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from extractor.config import settings

_bearer = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Reject requests without a valid Bearer token.

    If settings.api_key is empty, auth is disabled (development mode).
    """
    if not settings.api_key:
        return
    if credentials is None or credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
