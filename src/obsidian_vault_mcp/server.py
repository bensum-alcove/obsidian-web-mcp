"""Obsidian Vault MCP Server.

Exposes read/write access to an Obsidian vault over Streamable HTTP.
Designed to run behind Cloudflare Tunnel for secure remote access.
"""

import asyncio
import atexit
import contextlib
import json
import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .config import VAULT_MCP_PORT, VAULT_MCP_TOKEN, VAULT_PATH
from .frontmatter_index import FrontmatterIndex


class SecretPathMiddleware(BaseHTTPMiddleware):
    """Block /mcp requests unless they include the MCP_SECRET_PATH segment.
    If MCP_SECRET_PATH is not set, all requests are allowed (backward compatible).
    OAuth endpoints are always allowed.
    """
    async def dispatch(self, request, call_next):
        secret = os.environ.get("MCP_SECRET_PATH", "")
        if secret:
            path = request.url.path
            if path.startswith("/mcp") and not path.startswith(f"/mcp/{secret}"):
                return Response("Forbidden", status_code=403)
            if path.startswith(f"/mcp/{secret}"):
                new_path = "/mcp" + path[len(f"/mcp/{secret}"):]
                request.scope["path"] = new_path
        return await call_next(request)


logger = logging.getLogger(__name__)

# Global frontmatter index instance
frontmatter_index = FrontmatterIndex()
_semantic_started = False

atexit.register(frontmatter_index.stop)


@asynccontextmanager
async def lifespan(server):
    global _semantic_started
    frontmatter_index.start()
    if SEMANTIC_AVAILABLE and not _semantic_started:
        _semantic_started = True
        asyncio.create_task(startup_then_periodic())
    yield {"frontmatter_index": frontmatter_index}
    # Observer lives for process lifetime; atexit handles cleanup


# Create the MCP server
mcp = FastMCP(
    "obsidian_web_mcp",
    stateless_http=True,
    json_response=True,
    lifespan=lifespan,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "127.0.0.1:*",
            "localhost:*",
            "[::1]:*",
"vault.bensum.org",
            "vault-cb.bensum.org",
            "vault-alcove.bensum.org",
        ],
    ),
)


# --- Register all tools ---

from .tools.read import vault_read as _vault_read, vault_batch_read as _vault_batch_read, vault_read_section as _vault_read_section
from .tools.write import vault_write as _vault_write, vault_batch_frontmatter_update as _vault_batch_frontmatter_update, vault_patch_section as _vault_patch_section, vault_append as _vault_append, vault_batch_write as _vault_batch_write, vault_str_replace as _vault_str_replace, vault_batch_str_replace as _vault_batch_str_replace
from .tools.search import vault_search as _vault_search, vault_search_frontmatter as _vault_search_frontmatter
from .tools.manage import vault_list as _vault_list, vault_move as _vault_move, vault_delete as _vault_delete, vault_batch_delete as _vault_batch_delete, vault_recent_changes as _vault_recent_changes, vault_stats as _vault_stats, vault_session_start as _vault_session_start
from .tools.semantic_search import SEMANTIC_AVAILABLE, startup_then_periodic, vault_semantic_search as _vault_semantic_search, schedule_reindex as _schedule_reindex, vault_read_smart as _vault_read_smart
from .tools.context import vault_client_context as _vault_client_context
from .tools.entity import vault_entity as _vault_entity
from .tools.query import vault_query as _vault_query, vault_answer_context as _vault_answer_context
from .models import (
    VaultReadInput,
    VaultWriteInput,
    VaultBatchReadInput,
    VaultBatchFrontmatterUpdateInput,
    VaultSearchInput,
    VaultSearchFrontmatterInput,
    VaultListInput,
    VaultMoveInput,
    VaultDeleteInput,
    VaultPatchSectionInput,
    VaultAppendInput,
    VaultBatchWriteInput,
    VaultStrReplaceInput,
    VaultReadSectionInput,
    VaultBatchDeleteInput,
    VaultBatchStrReplaceInput,
    VaultRecentChangesInput,
    VaultStatsInput,
    VaultSessionStartInput,
    VaultReadSmartInput,
    VaultClientContextInput,
    VaultEntityInput,
    VaultQueryInput,
    VaultAnswerContextInput,
)


@mcp.tool(
    name="vault_read",
    description=(
        "Read a file from the Obsidian vault, returning content, metadata, and parsed YAML frontmatter. "
        "If the file has read_policy: section-only in its frontmatter, returns a preview and warning instead of full content — "
        "use vault_read_section for targeted reading, or pass force=True to read the full file regardless. "
        "Use max_chars to truncate large files."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_read(path: str, force: bool = False, max_chars: int | None = None) -> str:
    """Read a file from the vault."""
    inp = VaultReadInput(path=path, force=force, max_chars=max_chars)
    return _vault_read(inp.path, inp.force, inp.max_chars)


@mcp.tool(
    name="vault_batch_read",
    description=(
        "Read multiple files from the vault in one call. Handles missing files gracefully. "
        "Files with read_policy: section-only return a preview and warning per-file unless force=True is set."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_batch_read(paths: list[str], include_content: bool = True, force: bool = False) -> str:
    """Read multiple files at once."""
    inp = VaultBatchReadInput(paths=paths, include_content=include_content, force=force)
    return _vault_batch_read(inp.paths, inp.include_content, inp.force)


@mcp.tool(
    name="vault_write",
    description="Write a file to the Obsidian vault. Supports frontmatter merging with existing files. Creates parent directories by default.",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
def vault_write(path: str, content: str, create_dirs: bool = True, merge_frontmatter: bool = False) -> str:
    """Write a file to the vault."""
    inp = VaultWriteInput(path=path, content=content, create_dirs=create_dirs, merge_frontmatter=merge_frontmatter)
    result = _vault_write(inp.path, inp.content, inp.create_dirs, inp.merge_frontmatter)
    try:
        _schedule_reindex(inp.path)
    except Exception:
        pass
    return result


@mcp.tool(
    name="vault_batch_frontmatter_update",
    description="Update YAML frontmatter fields on multiple files without changing body content. Each update merges new fields into existing frontmatter.",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_batch_frontmatter_update(updates: list[dict]) -> str:
    """Batch update frontmatter fields."""
    inp = VaultBatchFrontmatterUpdateInput(updates=updates)
    result = _vault_batch_frontmatter_update(inp.updates)
    try:
        for item in inp.updates:
            _schedule_reindex(item.get("path", ""))
    except Exception:
        pass
    return result


@mcp.tool(
    name="vault_search",
    description="Search for text across vault files. Uses ripgrep if available, falls back to Python. Returns matching lines with context and frontmatter excerpts.",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_search(
    query: str,
    path_prefix: str | None = None,
    file_pattern: str = "*.md",
    max_results: int = 20,
    context_lines: int = 2,
) -> str:
    """Search vault file contents."""
    inp = VaultSearchInput(query=query, path_prefix=path_prefix, file_pattern=file_pattern, max_results=max_results, context_lines=context_lines)
    return _vault_search(inp.query, inp.path_prefix, inp.file_pattern, inp.max_results, inp.context_lines)


@mcp.tool(
    name="vault_search_frontmatter",
    description="Search vault files by YAML frontmatter field values. Queries an in-memory index for fast results. Supports exact match, contains, and field-exists queries.",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_search_frontmatter(
    field: str,
    value: str = "",
    match_type: str = "exact",
    path_prefix: str | None = None,
    max_results: int = 20,
) -> str:
    """Search by frontmatter fields."""
    inp = VaultSearchFrontmatterInput(field=field, value=value, match_type=match_type, path_prefix=path_prefix, max_results=max_results)
    return _vault_search_frontmatter(inp.field, inp.value, inp.match_type, inp.path_prefix, inp.max_results)


@mcp.tool(
    name="vault_list",
    description=(
        "List directory contents in the vault. Supports recursion depth, file/dir filtering, and glob patterns. "
        "Excludes .obsidian, .trash, .git directories. "
        "Pass frontmatter_fields to include specific YAML frontmatter values from each .md file in the listing."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_list(
    path: str = "",
    depth: int = 1,
    include_files: bool = True,
    include_dirs: bool = True,
    pattern: str | None = None,
    frontmatter_fields: list[str] | None = None,
) -> str:
    """List vault directory contents."""
    inp = VaultListInput(path=path, depth=depth, include_files=include_files, include_dirs=include_dirs, pattern=pattern, frontmatter_fields=frontmatter_fields)
    return _vault_list(inp.path, inp.depth, inp.include_files, inp.include_dirs, inp.pattern, inp.frontmatter_fields)


@mcp.tool(
    name="vault_move",
    description="Move a file or directory within the vault. Validates both source and destination paths.",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
def vault_move(source: str, destination: str, create_dirs: bool = True) -> str:
    """Move a file or directory."""
    inp = VaultMoveInput(source=source, destination=destination, create_dirs=create_dirs)
    result = _vault_move(inp.source, inp.destination, inp.create_dirs)
    try:
        _schedule_reindex(inp.source)
        _schedule_reindex(inp.destination)
    except Exception:
        pass
    return result


@mcp.tool(
    name="vault_delete",
    description="Delete a file by moving it to .trash/ in the vault root. Requires confirm=true as a safety gate. Does NOT hard delete.",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
def vault_delete(path: str, confirm: bool = False) -> str:
    """Delete a file (move to .trash/)."""
    inp = VaultDeleteInput(path=path, confirm=confirm)
    result = _vault_delete(inp.path, inp.confirm)
    try:
        _schedule_reindex(inp.path)
    except Exception:
        pass
    return result



@mcp.tool(
    name="vault_patch_section",
    description="Replace the content of a single markdown section without rewriting the entire file. Targets a heading and replaces everything between it and the next heading of the same or higher level.",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def vault_patch_section(path: str, section: str, content: str) -> str:
    """Patch a single markdown section in a vault file."""
    inp = VaultPatchSectionInput(path=path, section=section, content=content)
    result = _vault_patch_section(inp.path, inp.section, inp.content)
    try:
        _schedule_reindex(inp.path)
    except Exception:
        pass
    return result


@mcp.tool(
    name="vault_append",
    description="Append content to an existing vault file without reading or rewriting the whole file. Creates the file if it does not exist. Optionally inserts a blank-line separator before the new content.",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def vault_append(path: str, content: str, ensure_newline: bool = True) -> str:
    """Append content to a vault file."""
    inp = VaultAppendInput(path=path, content=content, ensure_newline=ensure_newline)
    result = _vault_append(inp.path, inp.content, inp.ensure_newline)
    try:
        _schedule_reindex(inp.path)
    except Exception:
        pass
    return result


@mcp.tool(
    name="vault_batch_write",
    description="Write up to 20 files in a single call. Each file is written atomically. Failures are reported per-file — the batch does not abort on a single error.",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
def vault_batch_write(files: list[dict]) -> str:
    """Write multiple vault files in one call."""
    inp = VaultBatchWriteInput(files=files)
    result = _vault_batch_write(inp.files)
    try:
        for item in inp.files:
            _schedule_reindex(item.get("path", ""))
    except Exception:
        pass
    return result

@mcp.tool(
    name="vault_str_replace",
    description=(
        "Replace a unique string in a vault file with another string. old_str must appear exactly once in the file. "
        "Safer and cheaper than vault_write for inline edits. "
        "Pass regex=True to treat old_str as a Python regex pattern (must match exactly once)."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def vault_str_replace(path: str, old_str: str, new_str: str, regex: bool = False) -> str:
    """Replace a unique string in a vault file."""
    inp = VaultStrReplaceInput(path=path, old_str=old_str, new_str=new_str, regex=regex)
    result = _vault_str_replace(inp.path, inp.old_str, inp.new_str, inp.regex)
    try:
        _schedule_reindex(inp.path)
    except Exception:
        pass
    return result


@mcp.tool(
    name="vault_read_section",
    description=(
        "Read a single markdown section by heading name. Returns only the content between the specified heading "
        "and the next heading of equal or higher level. Much cheaper than vault_read for large files — "
        "use this when you only need one section. "
        "If the section is not found, returns the list of available sections so you can retry."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_read_section(path: str, section: str) -> str:
    """Read a single markdown section by heading name."""
    inp = VaultReadSectionInput(path=path, section=section)
    return _vault_read_section(inp.path, inp.section)


@mcp.tool(
    name="vault_batch_delete",
    description=(
        "Delete multiple files in one call by moving them to .trash/. "
        "Requires confirm=true as a safety gate. "
        "Results reported per-file — the batch does not abort on individual failures."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
def vault_batch_delete(paths: list[str], confirm: bool = False) -> str:
    """Delete multiple files (move to .trash/)."""
    inp = VaultBatchDeleteInput(paths=paths, confirm=confirm)
    return _vault_batch_delete(inp.paths, inp.confirm)


@mcp.tool(
    name="vault_batch_str_replace",
    description=(
        "Replace unique strings in multiple files in one call. "
        "Each replacement specifies path, old_str, new_str, and an optional regex flag. "
        "Results reported per-file — the batch does not abort on individual failures. "
        "Safer and cheaper than multiple individual vault_str_replace calls."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def vault_batch_str_replace(replacements: list[dict]) -> str:
    """Replace unique strings in multiple vault files."""
    inp = VaultBatchStrReplaceInput(replacements=replacements)
    result = _vault_batch_str_replace(inp.replacements)
    try:
        for item in inp.replacements:
            _schedule_reindex(item.get("path", ""))
    except Exception:
        pass
    return result


@mcp.tool(
    name="vault_recent_changes",
    description=(
        "Return vault .md files modified after a given ISO datetime, sorted by most recent first. "
        "Useful for session-start: 'what changed since my last session?' "
        "Walks the full vault respecting excluded directories."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_recent_changes(since: str, limit: int = 20) -> str:
    """Return vault files modified after a given datetime."""
    inp = VaultRecentChangesInput(since=since, limit=limit)
    return _vault_recent_changes(inp.since, inp.limit)


@mcp.tool(
    name="vault_stats",
    description=(
        "Return vault-wide aggregate statistics: total .md files, total size in KB, "
        "10 largest files, 10 most recently modified files. "
        "Quick health overview without listing everything."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_stats() -> str:
    """Return vault-wide aggregate statistics."""
    VaultStatsInput()
    return _vault_stats()


@mcp.tool(
    name="vault_session_start",
    description=(
        "Bundle tool for session start. Returns vault stats, files modified since `since` (default 7 days), "
        "manifest summary (if present), and a pointer to _SCHEMA.md with the current write-rule count. "
        "One call replaces vault_stats + vault_recent_changes at session start."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_session_start(since: str | None = None) -> str:
    """Return bundled session-start data."""
    inp = VaultSessionStartInput(since=since)
    return _vault_session_start(inp.since)


@mcp.tool(
    name="vault_client_context",
    description=(
        "Scoped session-start for a single client. One call returns: the matched client note (full content + frontmatter), "
        "hot.md (Skills/hot.md), sp_folder_id from the client note frontmatter, and related file paths. "
        "Replaces vault_session_start + hot.md read + client search + client note read (4 calls → 1) for single-client tasks. "
        "If the client query matches multiple notes, returns candidates for disambiguation without reading content. "
        "If no match, returns client_note: null + template_path. "
        "Use vault_session_start for vault-wide or multi-client sessions."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_client_context(
    client: str,
    include_hot: bool = True,
    include_instructions: bool = False,
) -> str:
    """Return scoped session context for a single client."""
    inp = VaultClientContextInput(client=client, include_hot=include_hot, include_instructions=include_instructions)
    return _vault_client_context(inp.client, inp.include_hot, inp.include_instructions)


@mcp.tool(
    name="vault_entity",
    description=(
        "Look up a vault entity (client, team member, referral partner) by name or alias. "
        "Zero-LLM: reads the nightly-built _entities.json index. Single match returns the "
        "entity's file content plus its backlinks (path + line, capped at max_backlinks). "
        "Multiple matches return candidates for disambiguation, e.g. vault_entity(\"McGrath\") "
        "with several McGrath client files. No match returns the nearest names found."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_entity(name: str, max_backlinks: int = 15) -> str:
    """Look up a vault entity by name or alias."""
    inp = VaultEntityInput(name=name, max_backlinks=max_backlinks)
    return _vault_entity(inp.name, inp.max_backlinks)


@mcp.tool(
    name="vault_query",
    description=(
        "Fused hybrid search across the vault: merges the ripgrep keyword leg and the semantic "
        "embedding leg with Reciprocal Rank Fusion, then applies temporal decay (recent files rank "
        "higher; half-life varies by folder — Claude-Code-Prompts/ decays fastest, Clients/ slowest). "
        "Excludes _Archive/ and .trash/ by default. Each result includes the fused score, nearest "
        "heading, a chunk snippet, the file's updated date, a stale flag (unmodified >45 days), and "
        "an expand handle usable with vault_read_section. Use this instead of choosing between "
        "vault_search and vault_semantic_search."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_query(
    query: str,
    top_k: int = 8,
    path_prefix: str | None = None,
    include_archive: bool = False,
    decay: bool = True,
) -> str:
    """Fused hybrid search (RRF + temporal decay) across the vault."""
    inp = VaultQueryInput(query=query, top_k=top_k, path_prefix=path_prefix, include_archive=include_archive, decay=decay)
    return _vault_query(inp.query, inp.top_k, inp.path_prefix, inp.include_archive, inp.decay)


@mcp.tool(
    name="vault_answer_context",
    description=(
        "One-call brain-first pre-flight bundle: runs vault_query(question) and adds up to 3 relevant "
        "hot.md files (preferring ones sharing a top-level folder with the top results) plus a warnings "
        "list flagging stale or superseded results. Replaces a manual vault_query + hot.md read + "
        "staleness check sequence with a single call."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_answer_context(question: str, top_k: int = 6) -> str:
    """Brain-first pre-flight bundle: vault_query + hot.md + warnings."""
    inp = VaultAnswerContextInput(question=question, top_k=top_k)
    return _vault_answer_context(inp.question, inp.top_k)


if SEMANTIC_AVAILABLE:
    @mcp.tool(
        name="vault_semantic_search",
        description=(
            "Search vault files by semantic similarity rather than exact keywords. "
            "Returns ranked results with path, relevance score, snippet, and section heading. "
            "Best for conceptual or natural-language queries. "
            "Index refreshes on every write (debounced ~5s) and on a 30-minute floor. "
            "Current-session content is searchable within seconds. Use vault_search only for content written in the last few seconds."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    async def vault_semantic_search(
        query: str,
        max_results: int = 5,
        path_prefix: str | None = None,
    ) -> str:
        """Search vault by semantic similarity."""
        return await asyncio.to_thread(_vault_semantic_search, query, max_results, path_prefix)

    @mcp.tool(
        name="vault_read_smart",
        description=(
            "Read only the relevant sections of a large file using semantic similarity. "
            "Chunks the file by headings, embeds each chunk, and returns the top max_sections "
            "sections most relevant to your query. "
            "Best for large files (>8KB) like changelogs or master notes where you only need one topic. "
            "Falls back to full content if the file is small, or a section list if embedding fails."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
    async def vault_read_smart(path: str, query: str, max_sections: int = 3) -> str:
        """Read the most relevant sections of a file for a given query."""
        inp = VaultReadSmartInput(path=path, query=query, max_sections=max_sections)
        return await asyncio.to_thread(_vault_read_smart, inp.path, inp.query, inp.max_sections)


class TeamBotSiblingDispatcher:
    """Routes /mcp/teambot* to the teambot sub-app; all other requests to main app.

    The main app's middleware stack is completely untouched — a bug in the
    teambot route cannot affect the main vault-serving path.
    """

    def __init__(self, main_app, teambot_app):
        self._main = main_app
        self._teambot = teambot_app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "lifespan":
            await self._lifespan(scope, receive, send)
            return

        if scope.get("type") == "http":
            path = scope.get("path", "")
            if path == "/mcp/teambot" or path.startswith("/mcp/teambot/"):
                suffix = path[len("/mcp/teambot"):]
                new_scope = dict(scope)
                new_scope["path"] = "/mcp" + suffix
                new_scope["raw_path"] = ("/mcp" + suffix).encode()
                await self._teambot(new_scope, receive, send)
                return

        await self._main(scope, receive, send)

    async def _lifespan(self, scope, receive, send):
        """Drive both apps' lifespans by sending proper ASGI lifespan events to each."""
        import anyio

        asgi_version = scope.get("asgi", {})
        main_started = anyio.Event()
        teambot_started = anyio.Event()
        shutdown_trigger = anyio.Event()

        async def run_app_lifespan(app, started_event):
            """Simulate the full ASGI lifespan protocol for one app."""
            received_shutdown = anyio.Event()

            async def app_receive():
                if not started_event.is_set():
                    return {"type": "lifespan.startup"}
                await received_shutdown.wait()
                return {"type": "lifespan.shutdown"}

            async def app_send(message):
                if message["type"] == "lifespan.startup.complete":
                    started_event.set()
                elif message["type"] == "lifespan.startup.failed":
                    started_event.set()  # unblock even on failure

            # Drive lifespan in background; shutdown when outer trigger fires
            async with anyio.create_task_group() as sub_tg:
                sub_tg.start_soon(
                    app,
                    {"type": "lifespan", "asgi": asgi_version},
                    app_receive,
                    app_send,
                )
                await started_event.wait()
                await shutdown_trigger.wait()
                received_shutdown.set()

        try:
            await receive()  # consume lifespan.startup from uvicorn before proceeding
            async with anyio.create_task_group() as tg:
                tg.start_soon(run_app_lifespan, self._main, main_started)
                tg.start_soon(run_app_lifespan, self._teambot, teambot_started)
                await main_started.wait()
                await teambot_started.wait()
                await send({"type": "lifespan.startup.complete"})
                await receive()  # wait for uvicorn lifespan.shutdown
                shutdown_trigger.set()
            await send({"type": "lifespan.shutdown.complete"})
        except Exception as exc:
            msg = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            await send({"type": "lifespan.startup.failed", "message": msg})


def main():
    """Entry point. Run with streamable HTTP transport."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not VAULT_PATH.is_dir():
        logger.error(f"Vault path does not exist: {VAULT_PATH}")
        sys.exit(1)

    if not VAULT_MCP_TOKEN:
        logger.warning("VAULT_MCP_TOKEN is not set -- auth will reject all requests")

    # Build the Starlette app with auth middleware and OAuth endpoints
    try:
        from .auth import BearerAuthMiddleware
        import os as _os

        app = mcp.streamable_http_app()

        # /health is auth-exempt (see auth.py _AUTH_EXEMPT) but had no handler --
        # canary-restart verification needs a real 200 here.
        async def _health(request):
            return JSONResponse({"status": "ok"})

        app.routes.insert(0, Route("/health", _health, methods=["GET"]))

        # Mount OAuth 2.1 routes only when password gate is configured
        if _os.environ.get("VAULT_AUTH_PASSWORD"):
            from .oauth import oauth_routes
            for route in oauth_routes:
                app.routes.insert(0, route)
            logger.info("OAuth 2.1 password gate active")

        app.add_middleware(BearerAuthMiddleware)
        app.add_middleware(SecretPathMiddleware)  # outermost — runs first, blocks wrong paths before auth

        from .teambot import build_teambot_app
        teambot_app = build_teambot_app()
        combined = TeamBotSiblingDispatcher(app, teambot_app)

        logger.info(f"Starting server on port {VAULT_MCP_PORT} with bearer auth + OAuth + teambot route")

        import uvicorn
        uvicorn.run(
            combined,
            host="0.0.0.0",
            port=VAULT_MCP_PORT,
            log_level="info",
            proxy_headers=True,
            forwarded_allow_ips="*",
        )
    except Exception as e:
        logger.warning(f"Could not build app ({e}), falling back to mcp.run()")
        logger.warning("Auth will NOT be enforced in this mode")
        mcp.run(transport="streamable-http", port=VAULT_MCP_PORT)


if __name__ == "__main__":
    main()
