"""Teambot-scoped MCP sub-application for BS Team Bot.

Exposes a path-filtered subset of vault tools restricted to:
  - BS 2nd Brain/Alcove/Clients/
  - BS 2nd Brain/Alcove/Operations/Todo/
  - BS 2nd Brain/Alcove/Triage/

NOTE: This module must NOT import from .server (circular import risk).
"""

import json
import logging
import posixpath

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import config as _config
from .tools.read import vault_read as _vault_read, vault_batch_read as _vault_batch_read
from .tools.write import (
    vault_write as _vault_write,
    vault_append as _vault_append,
    vault_patch_section as _vault_patch_section,
    vault_str_replace as _vault_str_replace,
    vault_batch_write as _vault_batch_write,
    vault_batch_frontmatter_update as _vault_batch_frontmatter_update,
)
from .tools.search import vault_search as _vault_search
from .tools.manage import vault_list as _vault_list
from .tools.semantic_search import vault_semantic_search as _vault_semantic_search

logger = logging.getLogger(__name__)

TEAMBOT_ALLOWED_PREFIXES = [
    "BS 2nd Brain/Alcove/Clients",
    "BS 2nd Brain/Alcove/Operations/Todo",
    "BS 2nd Brain/Alcove/Triage",
]


def validate_teambot_path(path: str) -> None:
    """Raise ValueError if path is outside teambot allowed prefixes.

    Normalises ../  traversal before checking.
    """
    if not path or not path.strip():
        raise ValueError("Path must not be empty")

    normalized = posixpath.normpath(path)

    if normalized in (".", "..") or normalized.startswith("../"):
        raise ValueError(f"Path '{path}' resolves outside allowed scope")

    for prefix in TEAMBOT_ALLOWED_PREFIXES:
        if normalized == prefix or normalized.startswith(prefix + "/"):
            return

    raise ValueError(
        f"Path '{path}' is outside teambot scope. "
        f"Allowed prefixes: {TEAMBOT_ALLOWED_PREFIXES}"
    )


def resolve_semantic_prefix(path_prefix: str | None) -> list[str]:
    """Return the effective search prefix(es) for vault_semantic_search.

    Rules:
    - If path_prefix is within an allowed prefix, use it as-is.
    - If path_prefix is outside scope or not provided, fall back to all allowed
      prefixes (caller gets a union search across all in-scope roots).
    - Never returns any prefix outside TEAMBOT_ALLOWED_PREFIXES.
    """
    if path_prefix:
        normalized = posixpath.normpath(path_prefix)
        for allowed in TEAMBOT_ALLOWED_PREFIXES:
            if normalized == allowed or normalized.startswith(allowed + "/"):
                return [path_prefix]
    # Fallback: all allowed prefixes
    return list(TEAMBOT_ALLOWED_PREFIXES)


class TeamBotBearerAuthMiddleware(BaseHTTPMiddleware):
    """Validates Bearer tokens against TEAMBOT_MCP_TOKEN for the teambot route."""

    async def dispatch(self, request: Request, call_next):
        token = _config.TEAMBOT_MCP_TOKEN
        if not token:
            return JSONResponse(
                {"error": "Server misconfigured: TEAMBOT_MCP_TOKEN not set"},
                status_code=500,
            )

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "Missing or malformed Authorization header"},
                status_code=401,
            )

        if auth_header[7:] != token:
            return JSONResponse({"error": "Invalid token"}, status_code=401)

        return await call_next(request)


def build_teambot_app():
    """Build the teambot FastMCP sub-application with path-scoped tools.

    Returns a Starlette app that must be mounted at /mcp/teambot by the
    sibling dispatcher (requests arrive here with path rewritten to /mcp/*).
    """
    tb_mcp = FastMCP(
        "teambot_mcp",
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[
                "127.0.0.1:*",
                "localhost",
                "localhost:*",
                "[::1]:*",
                "vault.bensum.org",
            ],
        ),
    )

    @tb_mcp.tool(
        name="vault_read",
        description="Read a file from the vault (scoped: BS 2nd Brain/Alcove/Clients/, BS 2nd Brain/Alcove/Operations/Todo/, or BS 2nd Brain/Alcove/Triage/).",
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    def vault_read(path: str) -> str:
        try:
            validate_teambot_path(path)
        except ValueError as e:
            return json.dumps({"error": str(e), "path": path})
        return _vault_read(path)

    @tb_mcp.tool(
        name="vault_batch_read",
        description="Read multiple files (all paths must be in BS 2nd Brain/Alcove/Clients/, BS 2nd Brain/Alcove/Operations/Todo/, or BS 2nd Brain/Alcove/Triage/).",
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    def vault_batch_read(paths: list[str], include_content: bool = True) -> str:
        for p in paths:
            try:
                validate_teambot_path(p)
            except ValueError as e:
                return json.dumps({"error": str(e), "path": p})
        return _vault_batch_read(paths, include_content)

    @tb_mcp.tool(
        name="vault_write",
        description="Write a file (scoped: BS 2nd Brain/Alcove/Clients/, BS 2nd Brain/Alcove/Operations/Todo/, or BS 2nd Brain/Alcove/Triage/).",
        annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
    )
    def vault_write(path: str, content: str, create_dirs: bool = True, merge_frontmatter: bool = False) -> str:
        try:
            validate_teambot_path(path)
        except ValueError as e:
            return json.dumps({"error": str(e), "path": path})
        return _vault_write(path, content, create_dirs, merge_frontmatter)

    @tb_mcp.tool(
        name="vault_append",
        description="Append content to a vault file (scoped: BS 2nd Brain/Alcove/Clients/, BS 2nd Brain/Alcove/Operations/Todo/, or BS 2nd Brain/Alcove/Triage/).",
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
    )
    def vault_append(path: str, content: str, ensure_newline: bool = True) -> str:
        try:
            validate_teambot_path(path)
        except ValueError as e:
            return json.dumps({"error": str(e), "path": path})
        return _vault_append(path, content, ensure_newline)

    @tb_mcp.tool(
        name="vault_patch_section",
        description="Patch a markdown section (scoped: BS 2nd Brain/Alcove/Clients/, BS 2nd Brain/Alcove/Operations/Todo/, or BS 2nd Brain/Alcove/Triage/).",
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
    )
    def vault_patch_section(path: str, section: str, content: str) -> str:
        try:
            validate_teambot_path(path)
        except ValueError as e:
            return json.dumps({"error": str(e), "path": path})
        return _vault_patch_section(path, section, content)

    @tb_mcp.tool(
        name="vault_str_replace",
        description="Replace a unique string in a vault file (scoped: BS 2nd Brain/Alcove/Clients/, BS 2nd Brain/Alcove/Operations/Todo/, or BS 2nd Brain/Alcove/Triage/).",
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
    )
    def vault_str_replace(path: str, old_str: str, new_str: str) -> str:
        try:
            validate_teambot_path(path)
        except ValueError as e:
            return json.dumps({"error": str(e), "path": path})
        return _vault_str_replace(path, old_str, new_str)

    @tb_mcp.tool(
        name="vault_batch_write",
        description="Write multiple files (all paths must be in BS 2nd Brain/Alcove/Clients/, BS 2nd Brain/Alcove/Operations/Todo/, or BS 2nd Brain/Alcove/Triage/).",
        annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
    )
    def vault_batch_write(files: list[dict]) -> str:
        for f in files:
            p = f.get("path", "")
            try:
                validate_teambot_path(p)
            except ValueError as e:
                return json.dumps({"error": str(e), "path": p})
        return _vault_batch_write(files)

    @tb_mcp.tool(
        name="vault_batch_frontmatter_update",
        description="Update frontmatter on multiple files (all paths must be in BS 2nd Brain/Alcove/Clients/, BS 2nd Brain/Alcove/Operations/Todo/, or BS 2nd Brain/Alcove/Triage/).",
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    def vault_batch_frontmatter_update(updates: list[dict]) -> str:
        for u in updates:
            p = u.get("path", "")
            try:
                validate_teambot_path(p)
            except ValueError as e:
                return json.dumps({"error": str(e), "path": p})
        return _vault_batch_frontmatter_update(updates)

    @tb_mcp.tool(
        name="vault_search",
        description=(
            "Search vault files by keyword. path_prefix is required and must be "
            "within BS 2nd Brain/Alcove/Clients/, BS 2nd Brain/Alcove/Operations/Todo/, or BS 2nd Brain/Alcove/Triage/."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    def vault_search(
        query: str,
        path_prefix: str,
        file_pattern: str = "*.md",
        max_results: int = 20,
        context_lines: int = 2,
    ) -> str:
        try:
            validate_teambot_path(path_prefix)
        except ValueError as e:
            return json.dumps({"error": str(e)})
        return _vault_search(query, path_prefix, file_pattern, max_results, context_lines)

    @tb_mcp.tool(
        name="vault_list",
        description=(
            "List directory contents. path is required and must be "
            "within BS 2nd Brain/Alcove/Clients/, BS 2nd Brain/Alcove/Operations/Todo/, or BS 2nd Brain/Alcove/Triage/."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    def vault_list(
        path: str,
        depth: int = 1,
        include_files: bool = True,
        include_dirs: bool = True,
        pattern: str | None = None,
    ) -> str:
        try:
            validate_teambot_path(path)
        except ValueError as e:
            return json.dumps({"error": str(e)})
        return _vault_list(path, depth, include_files, include_dirs, pattern)

    @tb_mcp.tool(
        name="vault_semantic_search",
        description=(
            "Search vault files by semantic similarity (nickname/partial-name matching). "
            "path_prefix is optional: if within BS 2nd Brain/Alcove/Clients/, "
            "BS 2nd Brain/Alcove/Operations/Todo/, or BS 2nd Brain/Alcove/Triage/, "
            "that prefix is used. Otherwise (or if omitted), all three allowed prefixes "
            "are searched. Results are always restricted to the allowed prefixes."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    async def vault_semantic_search(
        query: str,
        max_results: int = 5,
        path_prefix: str | None = None,
    ) -> str:
        import asyncio
        prefixes = resolve_semantic_prefix(path_prefix)
        if len(prefixes) == 1:
            return await asyncio.to_thread(_vault_semantic_search, query, max_results, prefixes[0])
        # Union search across all allowed prefixes — merge and re-rank by score
        results_per_prefix = await asyncio.gather(
            *[asyncio.to_thread(_vault_semantic_search, query, max_results, p) for p in prefixes]
        )
        merged: list = []
        seen: set = set()
        for raw in results_per_prefix:
            try:
                parsed = json.loads(raw)
                items = parsed if isinstance(parsed, list) else parsed.get("results", [])
            except Exception:
                continue
            for item in items:
                key = item.get("path", "")
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return json.dumps(merged[:max_results])

    app = tb_mcp.streamable_http_app()
    app.add_middleware(TeamBotBearerAuthMiddleware)
    return app
