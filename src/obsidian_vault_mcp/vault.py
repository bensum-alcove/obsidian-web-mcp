"""Core filesystem operations for the Obsidian vault."""

import fnmatch
import hashlib
import logging
import os
import shutil
import stat
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from . import config
from . import mutation_ledger
from .write_contract import (
    PathMutationContext,
    WriteContext,
    WriteContractError,
    enforce as _enforce_write_contract,
    enforce_path_mutation as _enforce_path_mutation,
    get_mode as _write_contract_mode,
)

logger = logging.getLogger(__name__)


NEW_FILE_MODE = 0o604


# --------------------------------------------------------------------------
# Optimistic concurrency
#
# The revision token is a hash of a file's raw bytes on disk -- nothing else
# (no counter, no separate database) -- so it is process-restart-safe by
# construction and can never drift from the Markdown source of truth. A
# missing file has the sentinel revision "absent".
#
# Every mutation tool takes an *optional* `expected_revision`. Callers that
# never pass it get byte-identical legacy behaviour (unprotected write) --
# this is the compatibility story for callers that haven't migrated yet, and
# also the reason no separate off/shadow/enforce toggle is needed for
# *whether* a write is guarded: that's decided per-call by the caller, not a
# global switch. VAULT_OPTIMISTIC_CONCURRENCY_MODE exists only as an
# operator-controlled safety valve on top of that (see _concurrency_mode).
#
# Locking: a short-lived, per-resolved-path threading.Lock guards the
# read-current-bytes -> compare -> replace sequence so it is atomic with
# respect to other writers *in this process* (each vault is served by a
# single process). The lock is acquired fresh for each call and released
# before the call returns -- never held across a whole request or batch --
# so it cannot become the long-lived global lock the spec warns against.
# --------------------------------------------------------------------------

REVISION_ABSENT = "absent"


def compute_revision(data: bytes | None) -> str:
    """Compute a stable revision token from a file's raw bytes.

    ``data`` is None to represent a file that does not exist.
    """
    if data is None:
        return REVISION_ABSENT
    return "sha256:" + hashlib.sha256(data).hexdigest()


class RevisionConflictError(ValueError):
    """Raised when expected_revision does not match the file's current state.

    Subclasses ValueError so every existing tool's `except ValueError as e`
    handler still reports it as a structured error with no code changes at
    call sites that haven't been updated to catch it specifically.
    """

    def __init__(self, path: str, expected_revision: str, current_revision: str):
        self.path = path
        self.expected_revision = expected_revision
        self.current_revision = current_revision
        super().__init__(
            f"revision conflict for {path!r}: expected {expected_revision!r}, "
            f"current is {current_revision!r}. Reread the file, re-evaluate your "
            "change against the latest content, and reapply -- do not retry "
            "with this same stale content."
        )


def conflict_payload(e: RevisionConflictError) -> dict:
    """Machine-readable fields for a JSON error response, on top of "error"/path keys."""
    return {
        "conflict": True,
        "expected_revision": e.expected_revision,
        "current_revision": e.current_revision,
    }


def _concurrency_mode() -> str:
    """Read the optimistic-concurrency mode fresh on every call (see config.py).

    "off"     -- expected_revision is accepted but never checked (full revert).
    "shadow"  -- mismatches are detected and logged but never block a write.
    "enforce" -- mismatches raise RevisionConflictError (default).
    """
    mode = os.environ.get("VAULT_OPTIMISTIC_CONCURRENCY_MODE", "enforce").strip().lower()
    if mode not in ("off", "shadow", "enforce"):
        return "enforce"
    return mode


_path_locks: dict[Path, threading.Lock] = {}
_path_locks_guard = threading.Lock()


def _lock_for_path(path: Path) -> threading.Lock:
    with _path_locks_guard:
        lock = _path_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _path_locks[path] = lock
        return lock


_concurrency_metrics_guard = threading.Lock()
_concurrency_metrics = {"protected": 0, "unprotected": 0, "conflicted": 0}


def _record_metric(kind: str) -> None:
    with _concurrency_metrics_guard:
        _concurrency_metrics[kind] = _concurrency_metrics.get(kind, 0) + 1


def concurrency_metrics() -> dict:
    """Snapshot of protected/unprotected/conflicted write counts since process start."""
    with _concurrency_metrics_guard:
        return dict(_concurrency_metrics)


def _check_revision(
    relative_path: str,
    op: str,
    expected_revision: str | None,
    current_bytes: bytes | None,
    new_bytes: bytes | None = None,
) -> None:
    """Compare expected_revision against current on-disk state; raise on mismatch.

    ``new_bytes`` is the content a content-write is about to commit (None for
    move/delete, which have no "new content" to compare against). If the
    write would be a same-content no-op, it is allowed through even on a
    stale revision -- there is no data-loss risk in re-committing bytes that
    are already there.
    """
    if expected_revision is None:
        _record_metric("unprotected")
        return

    mode = _concurrency_mode()
    if mode == "off":
        _record_metric("unprotected")
        return

    actual_revision = compute_revision(current_bytes)
    is_noop = new_bytes is not None and current_bytes == new_bytes
    if actual_revision == expected_revision or is_noop:
        _record_metric("protected")
        return

    _record_metric("conflicted")
    would_block = mode == "enforce"
    logger.warning(
        "vault-concurrency[%s] op=%s path=%s expected=%s current=%s blocked=%s",
        mode, op, relative_path, expected_revision, actual_revision, would_block,
    )
    if would_block:
        raise RevisionConflictError(relative_path, expected_revision, actual_revision)


def resolve_vault_path(relative_path: str) -> Path:
    """Resolve a relative path against the vault root, with safety checks.

    Raises ValueError if the path escapes the vault, contains null bytes,
    or touches dotfile/dot-directory components.
    """
    if "\x00" in relative_path:
        raise ValueError("Path contains null bytes")

    # Check for dot-prefixed components (blocks .obsidian, .trash, dotfiles)
    parts = Path(relative_path).parts
    for part in parts:
        if part.startswith("."):
            raise ValueError(
                f"Path component '{part}' starts with '.'; dotfiles and hidden directories are not allowed"
            )

    resolved = (config.VAULT_PATH / relative_path).resolve()
    vault_root = config.VAULT_PATH.resolve()

    if not str(resolved).startswith(str(vault_root) + os.sep) and resolved != vault_root:
        raise ValueError("Path resolves outside the vault root")

    return resolved


def _iso_timestamp(ts: float) -> str:
    """Convert a Unix timestamp to an ISO 8601 string in UTC."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def read_file(relative_path: str) -> tuple[str, dict]:
    """Read a file and return (content, metadata).

    Metadata keys: size (int), modified (ISO str), created (ISO str), revision
    (str). Pass revision back as expected_revision to a mutation tool to
    guard against overwriting changes made since this read.
    """
    path = resolve_vault_path(relative_path)

    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {relative_path}")

    stat = path.stat()
    content = path.read_text(encoding="utf-8")

    # Revision is hashed from raw bytes, not the (possibly newline-normalised)
    # text above, so it always matches what write_file_atomic checks against.
    try:
        raw = path.read_bytes()
    except OSError:
        raw = content.encode("utf-8")

    metadata = {
        "size": stat.st_size,
        "modified": _iso_timestamp(stat.st_mtime),
        "created": _iso_timestamp(stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_ctime),
        "revision": compute_revision(raw),
    }

    return content, metadata


def write_file_atomic(
    relative_path: str,
    content: str,
    create_dirs: bool = True,
    tool: str = "",
    expected_revision: str | None = None,
    correlation_id: str | None = None,
    actor: str | None = None,
) -> tuple[bool, int]:
    """Write content to a file atomically.

    Returns (is_new_file, bytes_written). Writes to a tempfile in the same
    directory then replaces the target, so readers never see a partial write.

    Every mutation tool funnels through this one function, which is the
    write-contract gate's single enforcement point (see write_contract.py):
    a rejected write raises before any temp file is created, so the source
    file's bytes are never touched. The same is true of the optimistic-
    concurrency check: if expected_revision is given and doesn't match the
    current on-disk revision, this raises RevisionConflictError (or just
    logs, in "shadow" mode) before any temp file is created.

    The revision check and the write itself happen inside one short-lived
    per-path lock, so no other write_file_atomic call for the same path can
    interleave between the check and the replace.

    Every call also emits exactly one mutation-ledger event once the final
    result (success/reject/conflict) is known -- see mutation_ledger.py.
    correlation_id/actor are opaque passthrough fields recorded on that
    event when a caller has them; no current caller supplies them, so they
    are None in practice today.
    """
    encoded = content.encode("utf-8")
    old_hash_for_ledger: str | None = None
    try:
        if len(encoded) > config.MAX_CONTENT_SIZE:
            raise ValueError(
                f"Content size {len(encoded)} bytes exceeds limit of {config.MAX_CONTENT_SIZE} bytes"
            )

        path = resolve_vault_path(relative_path)

        with _lock_for_path(path):
            try:
                current_bytes = path.read_bytes()
            except FileNotFoundError:
                current_bytes = None
            old_hash_for_ledger = compute_revision(current_bytes)

            _check_revision(relative_path, tool, expected_revision, current_bytes, encoded)

            if _write_contract_mode() != "off":
                old_content_for_gate = None
                if current_bytes is not None:
                    try:
                        old_content_for_gate = current_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        old_content_for_gate = None
                _enforce_write_contract(
                    WriteContext(path=relative_path, old_content=old_content_for_gate, new_content=content, tool=tool)
                )

            try:
                original_mode = stat.S_IMODE(path.stat().st_mode)
                is_new = False
            except FileNotFoundError:
                original_mode = NEW_FILE_MODE
                is_new = True

            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            # Write to a temp file in the same directory, then atomic-replace.
            fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(encoded)
                    f.flush()
                    # Recheck immediately before committing. If an existing target was
                    # deleted while the temp file was being written, treat the commit as
                    # a new-file write instead of retaining stale metadata.
                    if not is_new:
                        try:
                            original_mode = stat.S_IMODE(path.stat().st_mode)
                        except FileNotFoundError:
                            original_mode = NEW_FILE_MODE
                            is_new = True
                    os.fchmod(f.fileno(), original_mode)
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
            except BaseException:
                # Clean up the temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            mutation_ledger.record(mutation_ledger.MutationEvent(
                tool=tool,
                path=relative_path,
                operation="create" if is_new else "update",
                result="success",
                old_hash=old_hash_for_ledger,
                new_hash=compute_revision(encoded),
                correlation_id=correlation_id,
                actor=actor,
            ))
            return is_new, len(encoded)
    except RevisionConflictError:
        mutation_ledger.record(mutation_ledger.MutationEvent(
            tool=tool,
            path=relative_path,
            operation="write",
            result="conflict",
            old_hash=old_hash_for_ledger,
            code="revision-conflict",
            correlation_id=correlation_id,
            actor=actor,
        ))
        raise
    except Exception as e:
        code = "write-contract-rejected" if isinstance(e, WriteContractError) else type(e).__name__
        mutation_ledger.record(mutation_ledger.MutationEvent(
            tool=tool,
            path=relative_path,
            operation="write",
            result="reject",
            old_hash=old_hash_for_ledger,
            code=code,
            correlation_id=correlation_id,
            actor=actor,
        ))
        raise


def move_path(
    source: str,
    destination: str,
    create_dirs: bool = True,
    expected_revision: str | None = None,
    correlation_id: str | None = None,
    actor: str | None = None,
) -> bool:
    """Move a file or directory from source to destination.

    Both paths are relative to the vault root. Raises if the destination
    already exists. expected_revision is only meaningful for files -- a
    directory has no single content hash to compare -- and raises ValueError
    if given for a directory source.

    Emits exactly one mutation-ledger event once the final result is known.
    """
    src = resolve_vault_path(source)
    dst = resolve_vault_path(destination)
    old_hash_for_ledger: str | None = None

    try:
        with _lock_for_path(src):
            if not src.exists():
                raise FileNotFoundError(f"Source does not exist: {source}")

            if dst.exists():
                raise FileExistsError(f"Destination already exists: {destination}")

            if expected_revision is not None and src.is_dir():
                raise ValueError("expected_revision is only supported for files, not directories")

            current_bytes = src.read_bytes() if src.is_file() else None
            old_hash_for_ledger = compute_revision(current_bytes)
            _check_revision(source, "move", expected_revision, current_bytes)

            if _write_contract_mode() != "off":
                _enforce_path_mutation(PathMutationContext(path=source, operation="move", destination=destination))

            if create_dirs:
                dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(src), str(dst))

            mutation_ledger.record(mutation_ledger.MutationEvent(
                tool="vault_move",
                path=source,
                operation="move",
                result="success",
                old_hash=old_hash_for_ledger,
                new_hash=old_hash_for_ledger,  # move never changes content
                destination=destination,
                correlation_id=correlation_id,
                actor=actor,
            ))
            return True
    except RevisionConflictError:
        mutation_ledger.record(mutation_ledger.MutationEvent(
            tool="vault_move",
            path=source,
            operation="move",
            result="conflict",
            old_hash=old_hash_for_ledger,
            code="revision-conflict",
            destination=destination,
            correlation_id=correlation_id,
            actor=actor,
        ))
        raise
    except Exception as e:
        code = "write-contract-rejected" if isinstance(e, WriteContractError) else type(e).__name__
        mutation_ledger.record(mutation_ledger.MutationEvent(
            tool="vault_move",
            path=source,
            operation="move",
            result="reject",
            old_hash=old_hash_for_ledger,
            code=code,
            destination=destination,
            correlation_id=correlation_id,
            actor=actor,
        ))
        raise


def delete_path(
    relative_path: str,
    expected_revision: str | None = None,
    correlation_id: str | None = None,
    actor: str | None = None,
) -> bool:
    """Soft-delete by moving the path into .trash/ at the vault root.

    Refuses to delete non-empty directories. expected_revision is only
    meaningful for files and raises ValueError if given for a directory.

    Emits exactly one mutation-ledger event once the final result is known.
    """
    path = resolve_vault_path(relative_path)
    old_hash_for_ledger: str | None = None

    try:
        with _lock_for_path(path):
            if not path.exists():
                raise FileNotFoundError(f"Path does not exist: {relative_path}")

            if path.is_dir() and any(path.iterdir()):
                raise ValueError(f"Refusing to delete non-empty directory: {relative_path}")

            if expected_revision is not None and path.is_dir():
                raise ValueError("expected_revision is only supported for files, not directories")

            current_bytes = path.read_bytes() if path.is_file() else None
            old_hash_for_ledger = compute_revision(current_bytes)
            _check_revision(relative_path, "delete", expected_revision, current_bytes)

            if _write_contract_mode() != "off":
                _enforce_path_mutation(PathMutationContext(path=relative_path, operation="delete"))

            trash_dir = config.VAULT_PATH.resolve() / ".trash"
            trash_dir.mkdir(exist_ok=True)

            dest = trash_dir / path.name

            # Avoid collisions in .trash by appending a timestamp
            if dest.exists():
                ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
                dest = trash_dir / f"{path.stem}_{ts}{path.suffix}"

            shutil.move(str(path), str(dest))

            mutation_ledger.record(mutation_ledger.MutationEvent(
                tool="vault_delete",
                path=relative_path,
                operation="delete",
                result="success",
                old_hash=old_hash_for_ledger,
                new_hash=None,
                correlation_id=correlation_id,
                actor=actor,
            ))
            return True
    except RevisionConflictError:
        mutation_ledger.record(mutation_ledger.MutationEvent(
            tool="vault_delete",
            path=relative_path,
            operation="delete",
            result="conflict",
            old_hash=old_hash_for_ledger,
            code="revision-conflict",
            correlation_id=correlation_id,
            actor=actor,
        ))
        raise
    except Exception as e:
        code = "write-contract-rejected" if isinstance(e, WriteContractError) else type(e).__name__
        mutation_ledger.record(mutation_ledger.MutationEvent(
            tool="vault_delete",
            path=relative_path,
            operation="delete",
            result="reject",
            old_hash=old_hash_for_ledger,
            code=code,
            correlation_id=correlation_id,
            actor=actor,
        ))
        raise


def list_directory(
    relative_path: str,
    depth: int = 1,
    include_files: bool = True,
    include_dirs: bool = True,
    pattern: str | None = None,
) -> list[dict]:
    """List directory contents recursively up to *depth* levels.

    Returns a list of dicts with keys: name, path (relative to vault),
    type ("file" or "dir"), size, modified.
    """
    depth = min(depth, config.MAX_LIST_DEPTH)

    root = resolve_vault_path(relative_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {relative_path}")

    vault_root = config.VAULT_PATH.resolve()
    results: list[dict] = []

    def _walk(dir_path: Path, current_depth: int) -> None:
        if current_depth > depth:
            return

        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: p.name.lower())
        except PermissionError:
            return

        for entry in entries:
            # Skip excluded directories at every level
            if entry.name in config.EXCLUDED_DIRS:
                continue

            is_dir = entry.is_dir()

            if is_dir and not include_dirs:
                # Still recurse even if we're not listing dirs
                _walk(entry, current_depth + 1)
                continue

            if not is_dir and not include_files:
                continue

            # Apply glob pattern filter
            if pattern and not fnmatch.fnmatch(entry.name, pattern):
                if is_dir:
                    _walk(entry, current_depth + 1)
                continue

            try:
                stat = entry.stat()
            except OSError:
                continue

            rel = str(entry.relative_to(vault_root))

            results.append({
                "name": entry.name,
                "path": rel,
                "type": "dir" if is_dir else "file",
                "size": stat.st_size,
                "modified": _iso_timestamp(stat.st_mtime),
            })

            if is_dir:
                _walk(entry, current_depth + 1)

    _walk(root, 1)
    return results
