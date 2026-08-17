"""Process-shared per-path mutation lock (vault-integrity-and-bo-authoring-remediation-v1).

``vault.py``'s ``_lock_for_path`` ``threading.Lock`` only serialises writers
inside this one server process. A conflicting writer in a different process
(a maintenance cron script, another CLI invocation, a second copy of this
server) is completely invisible to a ``threading.Lock`` -- it is in-memory,
per-process state. This module adds a real process-shared lock via
``fcntl.flock``, keyed by the resolved absolute path being mutated, so any
OTHER process that also acquires this same lock before touching that path is
correctly serialised against this server's writes.

The lock file itself deliberately does NOT live inside the vault. The live
checkout (``/mnt/c/Users/Ben Sum/obsidian-web-mcp``) serves vault content
mounted through WSL2's DrvFS bridge to the Windows filesystem, where
``flock()`` semantics on a Windows-mounted path are not a well-established
guarantee. Lock files instead live on the native Linux filesystem (default:
a directory under ``tempfile.gettempdir()``), which does not change what
*data* is being protected -- only where the lock bookkeeping itself lives.

TRUST BOUNDARY -- read this before assuming it closes every external-writer
race. This lock only serialises against another writer that ALSO acquires
the same lock file for the same resolved path before writing -- exactly the
"known/cooperating writer" framing used in this build's spec. It cannot and
does not protect against an arbitrary process that mutates vault content
without using this primitive (or the MCP write path in-process, which uses
it internally). As of this build, several known same-box, repo-controlled
scripts write directly to vault content without going through this lock --
see the build's output log for the inventory and the explicit,
carried-forward decision not to migrate them in this build (they live in
checkouts/repos outside this build's authorized scope). That is a real,
disclosed gap, not a claim of blanket protection against arbitrary writers.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import tempfile
from pathlib import Path

_LOCK_ROOT_ENV = "VAULT_MUTATION_LOCK_DIR"


def lock_root() -> Path:
    override = os.environ.get(_LOCK_ROOT_ENV, "").strip()
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "obsidian-vault-mcp-locks"


def _lock_file_for(resolved_path: Path) -> Path:
    digest = hashlib.sha256(str(resolved_path).encode("utf-8")).hexdigest()
    root = lock_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{digest}.lock"


@contextlib.contextmanager
def path_lock(resolved_path: Path):
    """Acquire a process-shared exclusive lock for one resolved filesystem path.

    Blocks until acquired. Safe to nest under vault.py's per-path
    ``threading.Lock`` (acquire the in-process lock first, then this one):
    ``flock`` exclusivity is per open-file-description, so separate threads
    or processes opening separate file descriptors against the same lock
    file correctly serialise against each other regardless of nesting order.
    """
    lock_path = _lock_file_for(resolved_path)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
