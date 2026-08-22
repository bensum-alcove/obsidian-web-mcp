"""Bearer token authentication middleware for the vault MCP server.

Two auth paths (when VAULT_AUTH_PASSWORD is set):
  1. Legacy: Authorization: Bearer <VAULT_MCP_TOKEN>
  2. OAuth:  Authorization: Bearer <oauth-issued token from /token>

When VAULT_AUTH_PASSWORD is NOT set, only path 1 is checked (existing behaviour).
"""

import hmac
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .config import VAULT_AUTH_PASSWORD, VAULT_BASE_URL, VAULT_MCP_TOKEN

TEAMBOT_MCP_TOKEN = os.environ.get("TEAMBOT_MCP_TOKEN", "")
_TEAMBOT_PREFIX = "/mcp/teambot"

# Paths that never require auth
_AUTH_EXEMPT = {
    "/health",
    "/authorize",
    "/token",
    "/.well-known/oauth-protected-resource",
    "/.well-known/oauth-protected-resource/mcp",
    "/.well-known/oauth-authorization-server",
}


def _get_base_url(request: Request) -> str:
    if VAULT_BASE_URL:
        return VAULT_BASE_URL.rstrip("/")
    return str(request.base_url).rstrip("/")


class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _AUTH_EXEMPT or path.startswith("/.well-known/"):
            return await call_next(request)

        # Teambot scoped route — separate token, rewrite path to /mcp before passing on
        if path.startswith(_TEAMBOT_PREFIX):
            auth_header = request.headers.get("Authorization", "")
            token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
            if TEAMBOT_MCP_TOKEN and token and hmac.compare_digest(token, TEAMBOT_MCP_TOKEN):
                request.scope["path"] = "/mcp" + path[len(_TEAMBOT_PREFIX):]
                return await call_next(request)
            return JSONResponse({"error": "Missing or invalid Authorization header"}, status_code=401)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

            # Path 1: legacy VAULT_MCP_TOKEN
            if VAULT_MCP_TOKEN and hmac.compare_digest(token, VAULT_MCP_TOKEN):
                return await call_next(request)

            # Path 2: OAuth-issued token (only active when password gate is set)
            if VAULT_AUTH_PASSWORD:
                from .oauth import is_valid_oauth_token
                if is_valid_oauth_token(token):
                    return await call_next(request)

        # Unauthorized — return appropriate 401
        if VAULT_AUTH_PASSWORD:
            base = _get_base_url(request)
            return Response(
                status_code=401,
                headers={
                    "WWW-Authenticate": (
                        f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource/mcp"'
                    )
                },
            )

        # No password gate — return simple JSON 401 (unchanged behaviour)
        if not VAULT_MCP_TOKEN:
            return JSONResponse(
                {"error": "Server misconfigured: no auth token set"},
                status_code=500,
            )
        return JSONResponse(
            {"error": "Missing or invalid Authorization header"},
            status_code=401,
        )
