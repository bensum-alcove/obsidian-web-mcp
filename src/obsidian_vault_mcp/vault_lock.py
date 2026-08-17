"""Process-shared per-path mutation lock (vault-integrity-and-bo-authority-remediation-v2).

``vault.py``'s ``_lock_for_path`` ``threading.Lock`` only serialises writers
inside this one server process. A conflicting writer in a different process
(a maintenance cron script, another CLI invocation, a second copy of this
server) is completely invisible to a ``threading.Lock`` -- it is in-memory,
per-process state. This module adds a real process-shared lock via
``fcntl``, keyed by the resolved absolute path being mutated, so any OTHER
process that also acquires this same lock before touching that path is
correctly serialised against this server's writes.

The lock file itself deliberately does NOT live inside the vault. The live
checkout (``/mnt/c/Users/Ben Sum/obsidian-web-mcp``) serves vault content
mounted through WSL2's DrvFS bridge to the Windows filesystem, where
``flock()``/``fcntl()`` semantics on a Windows-mounted path are not a
well-established guarantee. Lock bookkeeping instead lives on the native
Linux filesystem (default: a directory under ``tempfile.gettempdir()``),
which does not change what *data* is being protected -- only where the lock
bookkeeping itself lives.

BOUNDED REGISTRY (v2) -- codex-review-phase2-write-integrity-v2's LOW finding:
the v1 design created one lock file per ever-seen resolved path and never
removed them (999 files after one test run, unbounded growth). This version
uses exactly ONE persistent registry file, per-vault-lock-root, with
``LOCK_REGISTRY_SLOTS`` fixed byte-range slots (POSIX ``fcntl`` record
locks via ``fcntl.lockf``) -- a path hashes into one slot and locks that one
byte. Two distinct paths that hash to the same slot serialise against each
other (a false conflict that costs latency, never correctness); the
registry file itself never grows past its fixed slot count and is never
written with actual byte content (POSIX allows locking byte ranges beyond
EOF), so there is nothing to clean up and no unlink-on-release race to get
wrong.

Because POSIX ``fcntl`` record locks are associated with (inode, process)
rather than (fd): closing ANY file descriptor a process holds open on a
given file drops EVERY lock that process holds on that file, even ones
taken through a different fd. This module keeps exactly one persistent fd
per process for the registry file's whole lifetime (opened lazily, closed
only at process exit) so concurrent ``path_lock`` calls in the same process
-- even on different slots, even from different threads -- always lock
through that one fd and never race each other's byte ranges away.

IMPORTANT, and unlike the old per-path ``flock()`` design: POSIX ``fcntl``
record locks belong to the (process, inode) pair, not to a thread or an
individual open file description. Two THREADS in the *same* process do not
block each other through this lock alone -- the process already "owns" the
byte range either way, so a second acquisition from another thread in that
same process succeeds immediately. This is why every real call site in this
package (``vault.py``'s ``write_file_atomic``/``move_path``/``delete_path``)
holds its own per-path ``threading.Lock`` (``_lock_for_path``) ALONGSIDE this
lock -- that threading.Lock is what serialises threads within one process;
``path_lock`` is only ever the cross-process guarantee. A caller that needs
both must provide its own in-process mutex too, exactly as ``vault.py`` does.

TIMEOUT (v2) -- codex-review-phase2-write-integrity-v2's LOW finding: v1's
``path_lock`` blocked on the flock forever. This version bounds acquisition
to ``lock_timeout_seconds()`` (default 30s, configurable via
``VAULT_MUTATION_LOCK_TIMEOUT_SECONDS``) and raises ``LockTimeoutError`` on
expiry -- before any read or write of the protected path has happened, so a
timeout always leaves the target's bytes/mtime untouched. Never blocks
indefinitely, never silently proceeds unlocked.

TRUST BOUNDARY -- read this before assuming it closes every external-writer
race. This lock only serialises against another writer that ALSO acquires
the same lock (or calls ``atomic_write`` below) for the same resolved path
before writing -- exactly the "known/cooperating writer" framing used in
this build's spec. It cannot and does not protect against an arbitrary
process that mutates vault content without using this primitive. As of this
build, every known same-box, repo-controlled canonical-Markdown writer
(``log_syncer.py``, ``hot-md-curate.py --apply``, ``vault-audit.py
--autofix``) has been migrated onto this exact module -- see the build's
output log for the full inventory and disposition of every other direct
writer found.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import os
import stat
import tempfile
import threading
import time
from pathlib import Path

_LOCK_ROOT_ENV = "VAULT_MUTATION_LOCK_DIR"
_LOCK_TIMEOUT_ENV = "VAULT_MUTATION_LOCK_TIMEOUT_SECONDS"

DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0
LOCK_REGISTRY_SLOTS = 4096
_POLL_INTERVAL_SECONDS = 0.02


class LockTimeoutError(TimeoutError):
    """Raised when a mutation lock could not be acquired within the
    configured timeout. The caller's source bytes are guaranteed unchanged
    -- this is always raised before any write (or any read of the content
    being protected) begins."""

    def __init__(self, resolved_path, timeout_seconds: float):
        self.resolved_path = str(resolved_path)
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"timed out after {timeout_seconds}s acquiring the mutation lock for "
            f"{resolved_path!r}; no read or write of the target was attempted"
        )


def lock_root() -> Path:
    override = os.environ.get(_LOCK_ROOT_ENV, "").strip()
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "obsidian-vault-mcp-locks"


def lock_timeout_seconds() -> float:
    raw = os.environ.get(_LOCK_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_LOCK_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_LOCK_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_LOCK_TIMEOUT_SECONDS


def _registry_path() -> Path:
    root = lock_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / "registry.lock"


def _slot_for(resolved_path: Path) -> int:
    digest = hashlib.sha256(str(resolved_path).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % LOCK_REGISTRY_SLOTS


_registry_fd_lock = threading.Lock()
_registry_fd: int | None = None
_registry_fd_path: Path | None = None


def _registry_fd_handle() -> int:
    """One persistent fd per process for the whole registry file's lifetime.

    See the module docstring: POSIX record locks are released when ANY fd a
    process holds on the file is closed, so this fd is opened once (lazily)
    and reused by every ``path_lock`` call in this process -- including
    concurrently, from different threads, on different byte ranges.
    """
    global _registry_fd, _registry_fd_path
    path = _registry_path()
    with _registry_fd_lock:
        if _registry_fd is None or _registry_fd_path != path:
            _registry_fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            _registry_fd_path = path
        return _registry_fd


@contextlib.contextmanager
def path_lock(resolved_path: Path, timeout_seconds: float | None = None):
    """Acquire a process-shared byte-range lock for one resolved path's slot.

    Blocks up to ``timeout_seconds`` (default: ``lock_timeout_seconds()``).
    Raises ``LockTimeoutError`` on expiry -- no read or write of the target
    happens in that case. Safe to nest under vault.py's per-path
    ``threading.Lock`` (acquire the in-process lock first, then this one).
    """
    if timeout_seconds is None:
        timeout_seconds = lock_timeout_seconds()

    fd = _registry_fd_handle()
    slot = _slot_for(resolved_path)
    deadline = time.monotonic() + timeout_seconds

    while True:
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, slot, os.SEEK_SET)
            break
        except OSError as e:
            if e.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            if time.monotonic() >= deadline:
                raise LockTimeoutError(resolved_path, timeout_seconds) from None
            time.sleep(_POLL_INTERVAL_SECONDS)

    try:
        yield
    finally:
        fcntl.lockf(fd, fcntl.LOCK_UN, 1, slot, os.SEEK_SET)


def atomic_write(
    resolved_path: Path,
    data: bytes,
    *,
    timeout_seconds: float | None = None,
    mode: int | None = None,
) -> None:
    """Lock + atomically replace one file's content -- the single shared
    mutation primitive for same-box direct writers that are not the MCP
    server itself (``log_syncer.py``, ``hot-md-curate.py --apply``,
    ``vault-audit.py --autofix``). Creates parent directories if needed.

    Writes to a tempfile in the same directory, fsyncs, then ``os.replace``
    -- readers (including a concurrent MCP read) never observe a partial
    write. The whole sequence runs inside ``path_lock`` for this path, so it
    correctly serialises against the MCP server's own ``write_file_atomic``
    (which acquires the identical lock for the identical resolved path).
    """
    resolved_path = Path(resolved_path)
    with path_lock(resolved_path, timeout_seconds=timeout_seconds):
        if mode is None:
            try:
                mode = stat.S_IMODE(resolved_path.stat().st_mode)
            except FileNotFoundError:
                mode = 0o604

        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=resolved_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fchmod(f.fileno(), mode)
                os.fsync(f.fileno())
            os.replace(tmp_path, resolved_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def atomic_append(resolved_path: Path, data: bytes, *, timeout_seconds: float | None = None) -> None:
    """Lock + read-current-bytes + append + atomically replace -- for
    append-only canonical vault content (e.g. ``hot-md-curate.py``'s
    hot-archive/ rotation target). A bare O_APPEND write is not guaranteed
    atomic for arbitrary-sized writes and gives a concurrent MCP reader no
    serialization against this process at all; reading the current bytes
    and replacing under the same lock ``atomic_write`` uses closes both
    gaps -- two concurrent appenders can never interleave their bytes.
    """
    resolved_path = Path(resolved_path)
    with path_lock(resolved_path, timeout_seconds=timeout_seconds):
        try:
            current = resolved_path.read_bytes()
        except FileNotFoundError:
            current = b""
        try:
            mode = stat.S_IMODE(resolved_path.stat().st_mode)
        except FileNotFoundError:
            mode = 0o604

        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=resolved_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(current + data)
                f.flush()
                os.fchmod(f.fileno(), mode)
                os.fsync(f.fileno())
            os.replace(tmp_path, resolved_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
