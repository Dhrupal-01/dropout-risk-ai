"""
Access control, first stage (owner decision).

Every request that writes (POST, PUT, PATCH, DELETE) must carry `Authorization: Bearer
<API_ADMIN_TOKEN>`, compared in constant time. Reads are public while PUBLIC_READ_ONLY is true and
need the same token otherwise. If no token is configured on the server, writes are refused.

This is one shared token, not per-mentor login. Do not load real student data until per-mentor
authentication exists.
"""

import hmac
from typing import Annotated, Optional

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.core.config import get_settings
from backend.app.core.errors import UnauthorizedError

WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

bearer_scheme = HTTPBearer(auto_error=False, description="Admin token (API_ADMIN_TOKEN)")


def authorize(
    request: Request,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)],
) -> None:
    """Router-level dependency: writes always need the admin token; reads only when not public."""
    settings = get_settings()
    if request.method not in WRITE_METHODS and settings.PUBLIC_READ_ONLY:
        return
    if settings.API_ADMIN_TOKEN is None:
        raise UnauthorizedError("API_ADMIN_TOKEN is not configured on the server, so this request is refused.")
    expected = settings.API_ADMIN_TOKEN.get_secret_value().encode("utf-8")
    supplied = credentials.credentials.encode("utf-8") if credentials else b""
    if not hmac.compare_digest(supplied, expected):
        raise UnauthorizedError("Missing or invalid admin token. Send 'Authorization: Bearer <API_ADMIN_TOKEN>'.")
