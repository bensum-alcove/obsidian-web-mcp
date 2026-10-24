"""Atomic publication of the BO authoring namespace, never a multi-file rename loop.

Linux renameat2(RENAME_EXCHANGE) is the commit point. Before it, the full
snapshot is hidden; after it, all specs, schedule, and receipt are visible.
There is deliberately no weaker fallback on unsupported filesystems. The
existing mutation registry excludes old cooperating path writers as well as
new ones. Uncooperating/in-place external writers remain outside that lock's
trust boundary; a manifest check detects changes observed before exchange.
"""
from __future__ import annotations

import ctypes
import errno
import json
import logging
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Callable

from . import bo_contract, bo_guard, config, mutation_ledger, vault_lock
from . import vault
from .write_contract import WriteContext

logger = logging.getLogger(__name__)
NAMESPACE = "Personal/Build Orchestrator"
_commit_hooks: list[Callable[[list[str]], None]] = []


def register_commit_hook(hook):
    if hook not in _commit_hooks:
        _commit_hooks.append(hook)


def _sync_dir(path: Path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _exchange(left: Path, right: Path):
    libc = ctypes.CDLL(None, use_errno=True)
    fn = getattr(libc, "renameat2", None)
    if fn is None:
        raise OSError(errno.ENOTSUP, "atomic directory exchange is unavailable")
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    if fn(-100, os.fsencode(left), -100, os.fsencode(right), 2):
        error = ctypes.get_errno()
        raise OSError(error, "atomic authoring exchange failed; no fallback", str(left))


def _manifest(root: Path):
    result = {}
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError("authoring namespace contains a symlink or special file")
        result[str(path.relative_to(root))] = (
            info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode,
        )
    return result


def _stage_file(path: Path, data: bytes, mode: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Snapshot files are hardlinks. Never truncate them: replace the link.
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".stage-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def receipt_path(request_key: str) -> Path:
    return config.VAULT_PATH / NAMESPACE / ".authoring-receipts" / f"{request_key}.json"


def read_receipt(request_key: str, payload_hash: str) -> dict | None:
    try:
        record = json.loads(receipt_path(request_key).read_text())
    except FileNotFoundError:
        return None
    if record.get("payload_hash") != payload_hash:
        raise ValueError("request_id_payload_mismatch")
    return record["result"]


def commit(files: dict[str, str], *, schedule_path: str, expected_revision: str,
           tool: str, result: dict, receipt: tuple[str, str] | None = None,
           dry_run: bool = False) -> list[str]:
    """Validate the exact staged bytes and atomically publish, or write nothing.

    Caller holds registry_lock through graph preparation, so concurrent calls
    prepare against the result of their predecessor, rather than losing rows.
    Dry runs exercise all the same guard checks without staging or receipts.
    """
    root = vault.resolve_vault_path(NAMESPACE)
    if root != config.VAULT_PATH.resolve() / NAMESPACE:
        raise ValueError("authoring namespace must not be a symlink alias")
    originals = {}
    contexts = []
    for relative, content in files.items():
        path = vault.resolve_vault_path(relative)
        if not path.is_relative_to(root) or path == root or path != config.VAULT_PATH.resolve() / relative:
            raise ValueError("transaction path is outside the canonical authoring namespace")
        encoded = content.encode("utf-8")
        if len(encoded) > config.MAX_CONTENT_SIZE:
            raise ValueError("authoring file exceeds MAX_CONTENT_SIZE")
        current = path.read_bytes() if path.exists() else None
        if relative != schedule_path and current is not None:
            raise ValueError(f"refusing to overwrite existing spec at {relative!r}")
        if relative == schedule_path and vault.compute_revision(current) != expected_revision:
            raise vault.RevisionConflictError(relative, expected_revision, vault.compute_revision(current))
        originals[relative] = current
        contexts.append(WriteContext(path=relative, old_content=current.decode() if current is not None else None,
                                     new_content=content, tool=tool))

    # Full graph preparation already validated the complete resulting graph.
    # The ordinary write gates still run, with a private view of staged specs.
    staged = {p: c for p, c in files.items() if p.startswith(bo_guard.SPECS_PREFIX)}
    ids = [Path(p).stem for p in staged]
    with bo_contract.validation_batch(ids), bo_guard.staged_specs(staged):
        for ctx in contexts:
            if vault._write_contract_mode() != "off":
                vault._enforce_write_contract(ctx)
            if bo_guard.get_mode() != "off":
                bo_guard.enforce(ctx)
    if dry_run:
        return []

    baseline = _manifest(root)
    # A hidden sibling of the vault is outside ALL vault read/search/index
    # paths, including Python search fallbacks that walk hidden directories.
    # A writable same-filesystem parent is required; fail closed otherwise.
    stage = Path(tempfile.mkdtemp(dir=config.VAULT_PATH.resolve().parent,
                                  prefix=f".{config.VAULT_PATH.name}-bo-authoring-stage-"))
    exchanged = False
    warnings = []
    try:
        shutil.copytree(root, stage, dirs_exist_ok=True, copy_function=os.link)
        for relative, content in files.items():
            source = config.VAULT_PATH / relative
            mode = stat.S_IMODE(source.stat().st_mode) if source.exists() else vault.NEW_FILE_MODE
            _stage_file(stage / Path(relative).relative_to(NAMESPACE), content.encode(), mode)
        if receipt:
            key, payload_hash = receipt
            _stage_file(stage / ".authoring-receipts" / f"{key}.json",
                        json.dumps({"payload_hash": payload_hash, "result": result}).encode(), 0o600)
        for directory, _, _ in os.walk(stage, topdown=False):
            _sync_dir(Path(directory))
        _sync_dir(root.parent)
        _sync_dir(stage.parent)
        if _manifest(root) != baseline:
            raise ValueError("authoring namespace changed during staging; refusing stale snapshot")
        # Also detect an uncooperating target writer immediately before swap.
        for relative, original in originals.items():
            target = config.VAULT_PATH / relative
            current = target.read_bytes() if target.exists() else None
            if current != original:
                raise vault.RevisionConflictError(relative, vault.compute_revision(original), vault.compute_revision(current))
        _exchange(root, stage)
        exchanged = True
        try:
            _sync_dir(root.parent)
            _sync_dir(stage.parent)
        except BaseException:
            _exchange(root, stage)  # restore the entire snapshot, never a partial rollback
            exchanged = False
            _sync_dir(root.parent)
            _sync_dir(stage.parent)
            raise
    finally:
        # After a crash, an abandoned hidden snapshot is inert, never ingested.
        # Cleanup failure after a durable commit must not misreport activation.
        try:
            shutil.rmtree(stage)
        except OSError:
            if not exchanged:
                raise
            warnings.append("authoring snapshot cleanup pending")
    for relative, content in files.items():
        mutation_ledger.record(mutation_ledger.MutationEvent(
            tool=tool, path=relative, operation="create" if originals[relative] is None else "update",
            result="success", old_hash=vault.compute_revision(originals[relative]),
            new_hash=vault.compute_revision(content.encode()),
        ))
    for hook in _commit_hooks:
        try:
            hook(list(files))
        except Exception:
            logger.exception("authoring committed; index refresh failed")
            warnings.append("authoring committed; index refresh needs retry")
    return warnings
