"""Core filesystem operations for the Obsidian vault."""

import contextlib
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

from . import bo_guard
from . import config
from . import mutation_ledger
from . import vault_lock
from .bo_guard import BOGuardError
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


def canonical_vault_relative(resolved: Path) -> str:
    """Convert an already-resolved absolute path (from resolve_vault_path)
    back to its canonical, vault-relative, forward-slash string form.

    This is the value that must be handed to bo_guard/write_contract's
    WriteContext/PathMutationContext -- never the caller's raw path string.
    resolve_vault_path's ``.resolve()`` already collapsed `.`, `..`, `//` and
    any existing symlinks (including a vault-internal symlinked directory);
    reusing its output here means the guard evaluates the exact identity the
    write actually lands at, so an alias like
    'Personal/Build Orchestrator//schedules/x.yaml' or './Personal/...'
    cannot present a different path to the guard than the one it mutates
    (BL-1, opus-review-bo-authoring-contract-v4).
    """
    vault_root = config.VAULT_PATH.resolve()
    return str(resolved.relative_to(vault_root)).replace(os.sep, "/")


def _iso_timestamp(ts: float) -> str:
    """Convert a Unix timestamp to an ISO 8601 string in UTC."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def read_file(relative_path: str) -> tuple[str, dict]:
    """Read a file and return (content, metadata).

    Metadata keys: size (int), modified (ISO str), created (ISO str), revision
    (str). Pass revision back as expected_revision to a mutation tool to
    guard against overwriting changes made since this read.

    content and metadata["revision"] are derived from one single raw-bytes
    read on one open file descriptor -- never two separate filesystem calls.
    A writer landing between a stat()/read_text() pair and a later
    read_bytes() call used to be able to produce an incoherent (old content,
    new revision) pair; there is no second syscall here for a writer to land
    between.
    """
    path = resolve_vault_path(relative_path)

    try:
        fd = os.open(path, os.O_RDONLY)
    except (FileNotFoundError, NotADirectoryError):
        raise FileNotFoundError(f"Not a file: {relative_path}")

    with os.fdopen(fd, "rb") as f:
        file_stat = os.fstat(f.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            raise FileNotFoundError(f"Not a file: {relative_path}")
        raw = f.read()

    content = raw.decode("utf-8")

    metadata = {
        "size": file_stat.st_size,
        "modified": _iso_timestamp(file_stat.st_mtime),
        "created": _iso_timestamp(
            file_stat.st_birthtime if hasattr(file_stat, "st_birthtime") else file_stat.st_ctime
        ),
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
    per-path lock (both an in-process threading.Lock and a process-shared
    flock -- see vault_lock.py), so no other write_file_atomic call for the
    same path, and no other cooperating process, can interleave between the
    check and the replace. A second revision check immediately before the
    final os.replace() closes the remaining TOCTOU window against a writer
    that raced past the first check (see vault_lock.py's documented trust
    boundary for exactly which writers this does and doesn't cover).

    Every call also emits exactly one mutation-ledger event once the final
    result (success/reject/conflict) is known -- see mutation_ledger.py.
    correlation_id/actor are opaque passthrough fields recorded on that
    event when a caller has them; no current caller supplies them, so they
    are None in practice today.
    """
    encoded = content.encode("utf-8")
    old_hash_for_ledger: str | None = None
    intended_operation = "update"  # refined below once file existence is known; never the invalid "write"
    try:
        if len(encoded) > config.MAX_CONTENT_SIZE:
            raise ValueError(
                f"Content size {len(encoded)} bytes exceeds limit of {config.MAX_CONTENT_SIZE} bytes"
            )

        path = resolve_vault_path(relative_path)
        canonical_path = canonical_vault_relative(path)

        with _lock_for_path(path), vault_lock.path_lock(path):
            try:
                current_bytes = path.read_bytes()
            except FileNotFoundError:
                current_bytes = None
            old_hash_for_ledger = compute_revision(current_bytes)
            intended_operation = "create" if current_bytes is None else "update"

            _check_revision(relative_path, tool, expected_revision, current_bytes, encoded)

            old_content_for_gate = None
            if current_bytes is not None:
                try:
                    old_content_for_gate = current_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    old_content_for_gate = None
            # canonical_path (not the caller's raw relative_path) is what the
            # guard evaluates -- see canonical_vault_relative's docstring (BL-1).
            write_ctx = WriteContext(
                path=canonical_path, old_content=old_content_for_gate, new_content=content, tool=tool
            )
            if _write_contract_mode() != "off":
                _enforce_write_contract(write_ctx)
            if bo_guard.get_mode() != "off":
                bo_guard.enforce(write_ctx)

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

                    # Final on-disk revalidation at the activation boundary: this
                    # is the exact moment the Phase 2 review's probe injected an
                    # external write ("paused immediately before os.replace,
                    # wrote external ... then resumed"). Re-reading and
                    # re-checking here, still inside the same held locks, turns
                    # that race into an explicit RevisionConflictError instead of
                    # a silent overwrite for any caller that opted in via
                    # expected_revision (a no-op for legacy unprotected callers,
                    # exactly like the first _check_revision call).
                    try:
                        revalidate_bytes = path.read_bytes()
                    except FileNotFoundError:
                        revalidate_bytes = None
                    intended_operation = "create" if revalidate_bytes is None else "update"
                    _check_revision(relative_path, tool, expected_revision, revalidate_bytes, encoded)

                    # BO guard re-evaluation at the activation boundary
                    # (opus-review-bo-authoring-contract-v4, item 9). bo_guard's
                    # spec-rewrite check reads a REFERRING SCHEDULE's bytes (a
                    # different resolved path than `path`, so not covered by the
                    # lock held above) to revalidate the whole graph. Re-running
                    # the same enforce() call here, immediately before the
                    # commit, re-reads that schedule's current bytes and closes
                    # most of the window between the first check above and the
                    # os.replace() below. This narrows, but does not provably
                    # eliminate, the race: a writer to the referring schedule
                    # that lands in the few remaining lines between this check
                    # and os.replace() is not caught. No stronger guarantee is
                    # claimed -- see bo_guard.py's module docstring.
                    #
                    # LO5-2 (opus-review-bo-authoring-contract-v5): this second
                    # call used to reuse the ORIGINAL write_ctx, whose
                    # old_content was captured at the very first read, above --
                    # defeating the point of a second check for this path's own
                    # old_content (though not for the referring-schedule bytes,
                    # which _read_vault_text always re-reads fresh regardless).
                    # Rebuilt here from revalidate_bytes, the bytes just re-read
                    # immediately above, so both halves of the re-check see
                    # current data.
                    if bo_guard.get_mode() != "off":
                        revalidate_content_for_gate = None
                        if revalidate_bytes is not None:
                            try:
                                revalidate_content_for_gate = revalidate_bytes.decode("utf-8")
                            except UnicodeDecodeError:
                                revalidate_content_for_gate = None
                        fresh_write_ctx = WriteContext(
                            path=canonical_path, old_content=revalidate_content_for_gate,
                            new_content=content, tool=tool,
                        )
                        bo_guard.enforce(fresh_write_ctx)

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
            operation=intended_operation,
            result="conflict",
            old_hash=old_hash_for_ledger,
            code="revision-conflict",
            correlation_id=correlation_id,
            actor=actor,
        ))
        raise
    except vault_lock.LockTimeoutError:
        # No read or write of the target happened -- source bytes/mtime are
        # guaranteed unchanged. Ledgered as its own code, distinct from a
        # generic reject, so operators can tell a busy path apart from a
        # rejected write.
        mutation_ledger.record(mutation_ledger.MutationEvent(
            tool=tool,
            path=relative_path,
            operation=intended_operation,
            result="reject",
            old_hash=old_hash_for_ledger,
            code="lock-timeout",
            correlation_id=correlation_id,
            actor=actor,
        ))
        raise
    except Exception as e:
        code = (
            "write-contract-rejected" if isinstance(e, WriteContractError)
            else "bo-guard-rejected" if isinstance(e, BOGuardError)
            else type(e).__name__
        )
        mutation_ledger.record(mutation_ledger.MutationEvent(
            tool=tool,
            path=relative_path,
            operation=intended_operation,
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

    Source AND destination are locked together (in-process lock plus the
    process-shared flock from vault_lock.py) for the whole operation, in a
    deterministic order (sorted by resolved path) so two concurrent movers
    that both touch the same pair of paths can never deadlock. Because the
    destination is locked for the *entire* operation -- not just checked once
    up front -- two concurrent moves to the same destination are strictly
    serialised: whichever acquires the destination lock first completes and
    the other observes the now-existing destination and fails explicitly,
    with its own source left untouched. Path-contract validation runs against
    both source and destination.

    Emits exactly one mutation-ledger event once the final result is known.
    """
    src = resolve_vault_path(source)
    dst = resolve_vault_path(destination)
    canonical_source = canonical_vault_relative(src)
    canonical_destination = canonical_vault_relative(dst)
    old_hash_for_ledger: str | None = None

    # Deterministic lock ordering across the (up to) two distinct paths
    # involved avoids deadlock between two movers that both need locks on an
    # overlapping pair of paths (e.g. two sources racing to the same
    # destination) -- see module docstring / vault_lock.py.
    lock_targets = sorted({src, dst}, key=str)

    try:
        with contextlib.ExitStack() as stack:
            for target in lock_targets:
                stack.enter_context(_lock_for_path(target))
                stack.enter_context(vault_lock.path_lock(target))

            if not src.exists():
                raise FileNotFoundError(f"Source does not exist: {source}")

            if dst.exists():
                raise FileExistsError(f"Destination already exists: {destination}")

            if expected_revision is not None and src.is_dir():
                raise ValueError("expected_revision is only supported for files, not directories")

            current_bytes = src.read_bytes() if src.is_file() else None
            old_hash_for_ledger = compute_revision(current_bytes)
            _check_revision(source, "move", expected_revision, current_bytes)

            # canonical_source/canonical_destination (not the caller's raw
            # strings) are what the guard evaluates -- see
            # canonical_vault_relative's docstring (BL-1).
            move_ctx = PathMutationContext(
                path=canonical_source, operation="move", destination=canonical_destination
            )
            if _write_contract_mode() != "off":
                _enforce_path_mutation(move_ctx)
            if bo_guard.get_mode() != "off":
                bo_guard.enforce_path_mutation(move_ctx)

            if create_dirs:
                dst.parent.mkdir(parents=True, exist_ok=True)

            # Final destination-existence recheck at the activation boundary,
            # immediately before the filesystem mutation. Both locks for `dst`
            # have been held continuously since before the first existence
            # check above, so this is defence-in-depth against any code path
            # (create_dirs, symlink races) that could otherwise have
            # materialised something at `dst` in between.
            if dst.exists():
                raise FileExistsError(f"Destination already exists: {destination}")

            # Activation-boundary re-check (HI5-1, opus-review-bo-authoring-
            # contract-v5): bo_guard's directory-move validation enumerates
            # every nested file under `src` and reads each one to evaluate it
            # -- seconds of work for a large directory -- but this function
            # only ever called enforce_path_mutation ONCE, before that work,
            # while holding no lock on the individual nested files (only on
            # `src`/`dst` themselves). A concurrent write landing on a nested
            # file mid-enumeration reproduced 12/12 under real cooperating
            # processes: malformed, never-(re)validated bytes could still
            # land under BO authority. Re-running the same check here,
            # immediately before the filesystem mutation, re-enumerates and
            # re-reads every nested file's THEN-current bytes (no caching in
            # _files_under/_read_vault_text), closing the window down to the
            # few remaining lines between this call and shutil.move -- the
            # same pattern write_file_atomic already uses for its own
            # activation-boundary re-check.
            if bo_guard.get_mode() != "off":
                bo_guard.enforce_path_mutation(move_ctx)

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
    except vault_lock.LockTimeoutError:
        mutation_ledger.record(mutation_ledger.MutationEvent(
            tool="vault_move",
            path=source,
            operation="move",
            result="reject",
            old_hash=old_hash_for_ledger,
            code="lock-timeout",
            destination=destination,
            correlation_id=correlation_id,
            actor=actor,
        ))
        raise
    except Exception as e:
        code = (
            "write-contract-rejected" if isinstance(e, WriteContractError)
            else "bo-guard-rejected" if isinstance(e, BOGuardError)
            else type(e).__name__
        )
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
    canonical_path = canonical_vault_relative(path)
    old_hash_for_ledger: str | None = None

    try:
        with _lock_for_path(path), vault_lock.path_lock(path):
            if not path.exists():
                raise FileNotFoundError(f"Path does not exist: {relative_path}")

            if path.is_dir() and any(path.iterdir()):
                raise ValueError(f"Refusing to delete non-empty directory: {relative_path}")

            if expected_revision is not None and path.is_dir():
                raise ValueError("expected_revision is only supported for files, not directories")

            current_bytes = path.read_bytes() if path.is_file() else None
            old_hash_for_ledger = compute_revision(current_bytes)
            _check_revision(relative_path, "delete", expected_revision, current_bytes)

            # canonical_path (not the caller's raw relative_path) is what the
            # guard evaluates -- see canonical_vault_relative's docstring (BL-1).
            delete_ctx = PathMutationContext(path=canonical_path, operation="delete")
            if _write_contract_mode() != "off":
                _enforce_path_mutation(delete_ctx)
            if bo_guard.get_mode() != "off":
                bo_guard.enforce_path_mutation(delete_ctx)

            # Activation-boundary re-check (HI5-1), matching move_path's own
            # fix and write_file_atomic's established pattern -- re-reads/
            # re-enumerates fresh bytes immediately before the filesystem
            # mutation rather than trusting the single check done above.
            if bo_guard.get_mode() != "off":
                bo_guard.enforce_path_mutation(delete_ctx)

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
    except vault_lock.LockTimeoutError:
        mutation_ledger.record(mutation_ledger.MutationEvent(
            tool="vault_delete",
            path=relative_path,
            operation="delete",
            result="reject",
            old_hash=old_hash_for_ledger,
            code="lock-timeout",
            correlation_id=correlation_id,
            actor=actor,
        ))
        raise
    except Exception as e:
        code = (
            "write-contract-rejected" if isinstance(e, WriteContractError)
            else "bo-guard-rejected" if isinstance(e, BOGuardError)
            else type(e).__name__
        )
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
