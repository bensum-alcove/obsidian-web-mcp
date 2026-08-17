"""Append-only local audit ledger for vault mutations.

Records *metadata* about every content/path mutation (who/what/when/result)
after the final outcome is known -- never note bodies, never secret values,
never a second source of truth for vault content. Git (via the existing
periodic vault-backup commit job on each deployed vault) remains the
canonical history of *what changed*; this ledger answers a narrower,
faster question that git alone cannot: which *tool call* touched a path,
whether it was accepted/rejected/conflicted, and why -- indexed by wall-clock
time rather than by content diff. A "vault backup 2026-08-17 22:00" commit
tells you a file's bytes changed sometime in a 15-minute window; a ledger
query tells you it was `vault_write` at 22:03:41, accepted, with these two
content hashes. Use `git log -p <path>` for the actual diff; use this ledger
to find *which call* produced it, or to see rejected/conflicted attempts that
never touched the file at all (and therefore never appear in git).

Storage: one JSON object per line, appended to
``<ledger_dir>/mutations.jsonl`` via a stdlib ``RotatingFileHandler`` (size-
bounded, backupCount-bounded -- see VAULT_MUTATION_LEDGER_MAX_BYTES /
VAULT_MUTATION_LEDGER_BACKUP_COUNT in config.py). Default ledger_dir is
``<VAULT_PATH>/.mutation-ledger`` -- a dot-directory, so it is already
outside every read tool's reach: ``resolve_vault_path`` refuses any
dot-prefixed path component, and ``config.EXCLUDED_DIRS`` (which this module
extends with ``.mutation-ledger``) is what ``vault_list``, ``vault_search``,
``vault_semantic_search``, ``vault_recent_changes``, ``vault_stats``, and the
frontmatter index all already filter on. Overridable via
VAULT_MUTATION_LEDGER_DIR for an operator who wants it outside the vault
entirely.

Feature mode is one environment variable, VAULT_MUTATION_LEDGER_MODE
("on" default | "off"), read fresh on every call -- consistent with this
project's other toggles (VAULT_WRITE_CONTRACT_MODE, VAULT_OPTIMISTIC_
CONCURRENCY_MODE). Unlike those, there is no "shadow" state: a ledger event
either gets recorded or it doesn't, and recording never blocks or fails an
otherwise-valid mutation. record() catches everything internally and reports
failures only through health_metrics()["failed"] -- callers should surface
that counter as a health/metrics signal, not treat it as an error.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

_EVENTS_LOGGER_NAME = "obsidian_vault_mcp.mutation_ledger.events"
_events_logger = logging.getLogger(_EVENTS_LOGGER_NAME)
_events_logger.propagate = False  # ledger lines must never leak into the app's own log stream
_events_logger.setLevel(logging.INFO)

_handler_lock = threading.Lock()
_configured_path: Path | None = None

_metrics_lock = threading.Lock()
_metrics = {"recorded": 0, "failed": 0}


def _record_metric(kind: str) -> None:
    with _metrics_lock:
        _metrics[kind] = _metrics.get(kind, 0) + 1


def health_metrics() -> dict:
    """Snapshot of recorded/failed ledger-write counts since process start.

    A nonzero (and growing) "failed" count is the health-warning signal
    called for by the build spec's "ledger outage must not block writes"
    requirement -- writes keep succeeding either way, but an operator can
    alert on this counter.
    """
    with _metrics_lock:
        return dict(_metrics)


def _mode() -> str:
    mode = os.environ.get("VAULT_MUTATION_LEDGER_MODE", "on").strip().lower()
    if mode not in ("on", "off"):
        return "on"
    return mode


def _ledger_dir() -> Path:
    override = os.environ.get("VAULT_MUTATION_LEDGER_DIR", "").strip()
    if override:
        return Path(override)
    return config.VAULT_PATH / ".mutation-ledger"


def _max_bytes() -> int:
    try:
        return int(os.environ.get("VAULT_MUTATION_LEDGER_MAX_BYTES", "5000000"))
    except ValueError:
        return 5_000_000


def _backup_count() -> int:
    try:
        return int(os.environ.get("VAULT_MUTATION_LEDGER_BACKUP_COUNT", "10"))
    except ValueError:
        return 10


def _ensure_handler() -> logging.Handler | None:
    """Lazily (re)create the rotating file handler for the current ledger dir.

    Returns None if the handler cannot be created (e.g. permissions, a
    read-only filesystem) -- callers must treat that as "ledger unavailable"
    and bump the "failed" metric, never raise.
    """
    global _configured_path
    target_path = _ledger_dir() / "mutations.jsonl"

    with _handler_lock:
        if _configured_path == target_path and _events_logger.handlers:
            return _events_logger.handlers[0]

        for old_handler in list(_events_logger.handlers):
            _events_logger.removeHandler(old_handler)
            old_handler.close()

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                target_path,
                maxBytes=_max_bytes(),
                backupCount=_backup_count(),
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            _events_logger.addHandler(handler)
            _configured_path = target_path
            return handler
        except OSError:
            logger.exception("mutation-ledger: failed to open ledger file at %s", target_path)
            _configured_path = None
            return None


@dataclass(frozen=True)
class MutationEvent:
    """One vault-mutation attempt, metadata only -- never content or secrets."""

    tool: str
    path: str
    operation: str  # "create" | "update" | "delete" | "move"
    result: str  # "success" | "reject" | "conflict"
    old_hash: str | None = None
    new_hash: str | None = None
    code: str | None = None  # reject/conflict classifier, e.g. "revision-conflict"
    correlation_id: str | None = None  # threaded through if a caller supplies one; None otherwise
    actor: str | None = None  # threaded through if a caller supplies one; None otherwise
    destination: str | None = None  # move only

    def to_dict(self, timestamp: str) -> dict:
        return {
            "timestamp": timestamp,
            "actor": self.actor,
            "tool": self.tool,
            "path": self.path,
            "operation": self.operation,
            "destination": self.destination,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "result": self.result,
            "code": self.code,
            "correlation_id": self.correlation_id,
        }


def record(event: MutationEvent) -> None:
    """Best-effort append of one mutation event to the ledger.

    Never raises. A failure here must never block or fail an otherwise valid
    vault mutation -- the caller has already committed (or rejected) the
    actual write by the time this is called. Failures increment the
    "failed" health metric instead of propagating.
    """
    if _mode() == "off":
        return
    try:
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        handler = _ensure_handler()
        if handler is None:
            _record_metric("failed")
            return
        line = json.dumps(event.to_dict(timestamp), sort_keys=True, ensure_ascii=False)
        _events_logger.info(line)
        _record_metric("recorded")
    except Exception:
        logger.exception("mutation-ledger: failed to record event for tool=%s path=%s", event.tool, event.path)
        _record_metric("failed")


def _ledger_files(ledger_dir: Path) -> list[Path]:
    if not ledger_dir.exists():
        return []
    try:
        return [p for p in ledger_dir.glob("mutations.jsonl*") if p.is_file()]
    except OSError:
        return []


def query_events(
    ledger_dir: Path | None = None,
    path_prefix: str | None = None,
    tool: str | None = None,
    result: str | None = None,
    since: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Read-only incident-query path over the ledger: current file + rotated backups.

    Malformed lines are skipped rather than raising -- the ledger is
    diagnostic, not a strict data store. Returns events newest-first,
    filtered by the given criteria (all optional, AND-combined).
    """
    target_dir = ledger_dir if ledger_dir is not None else _ledger_dir()
    events: list[dict] = []

    for file_path in _ledger_files(target_dir):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

    if path_prefix:
        events = [e for e in events if str(e.get("path") or "").startswith(path_prefix)]
    if tool:
        events = [e for e in events if e.get("tool") == tool]
    if result:
        events = [e for e in events if e.get("result") == result]
    if since:
        events = [e for e in events if str(e.get("timestamp") or "") >= since]

    events.sort(key=lambda e: str(e.get("timestamp") or ""), reverse=True)
    return events[:limit]
