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
from dataclasses import dataclass
import errno
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import stat
import tempfile
import subprocess
from typing import Callable

import frontmatter

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
    info = root.stat()
    result = {".": (info.st_dev, info.st_ino, info.st_mode)}
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


class PreparationChanged(ValueError):
    """A read dependency changed; discard the entire preparation and retry."""


def _guard_inputs(build_ids=(), schedule_paths=()) -> str:
    """Fingerprint mutable adapter inputs without reading credential values.

    The adapter owns path resolution. Its read-only imports report only a
    digest, never configuration/DB contents. Only non-secret task columns
    consumed by the adapter's graph/rewrite checks are selected, for graph
    identities and rows bound to the affected schedules. Heartbeats, events
    and unrelated tasks cannot invalidate approval. Source, registry,
    project config and repository config remain bound; refs are not read by
    authoring validation and are deliberately excluded.
    This is deliberately not a cached DB approval or a new locking protocol.
    """
    directory = str(Path(bo_contract.resolve_authoring_contract_path()).resolve().parent)
    script = r"""
import sys,os,json,hashlib,subprocess,sqlite3
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import db,model_registry,project_resolve
root=Path(sys.argv[1]); paths=set(root.rglob('*.py'))
paths.update([root/'config.yaml',model_registry.registry_path(),model_registry._state_path(),model_registry.DEFAULT_PATH])
for project in (project_resolve.load_config().get('projects') or {}).values():
 repo=project.get('repo_dir')
 if not repo: continue
 for option in ('--git-dir','--git-common-dir'):
  p=subprocess.run(['git','-C',repo,'rev-parse',option],capture_output=True,text=True,timeout=2)
  if p.returncode: continue
  git=Path(p.stdout.strip());git=git if git.is_absolute() else Path(repo)/git
  paths.update(git/name for name in ('config','commondir'))
records=[]
for path in sorted(paths):
 try:
  st=path.stat(); value=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_mode)
 except FileNotFoundError:value=None
 records.append((str(path),value))
scope=json.loads(sys.argv[2]); ids=scope['ids']; schedules=scope['schedules']
# These are authority facts, not operational metadata or credential columns.
# Keep the adapter's validation semantics authoritative: a changed fact only
# discards preparation; it never grants permission or filters a refusal.
columns='id,status,source_schedule,spec_hash,schedule_hash,completion_contract_hash,cc_project,goal_id'
with sqlite3.connect(Path(db.DB_PATH).resolve().as_uri()+'?mode=ro',uri=True) as conn:
 conn.execute('PRAGMA query_only=ON')
 rows=conn.execute('SELECT '+columns+' FROM tasks WHERE id IN ('+','.join('?' for _ in ids)+') OR source_schedule IN ('+','.join('?' for _ in schedules)+') ORDER BY id',ids+schedules).fetchall()
print(hashlib.sha256(json.dumps([records,rows]).encode()).hexdigest())
"""
    response = subprocess.run(
        [config.BO_AUTHORING_CONTRACT_PYTHON, '-c', script, directory,
         json.dumps({'ids': sorted(build_ids), 'schedules': sorted(schedule_paths)})],
        capture_output=True, text=True, timeout=config.BO_AUTHORING_CONTRACT_TIMEOUT_SECONDS,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
    )
    digest = response.stdout.strip()
    if response.returncode or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('cannot bind authoring validation to mutable guard inputs')
    return digest


@dataclass(frozen=True)
class InputBinding:
    builds: tuple
    schedule_path: str | None
    signature: tuple

    def current(self):
        return capture_inputs(builds=self.builds, schedule_path=self.schedule_path)


def capture_inputs(*, builds=(), schedule_path=None):
    """Exact authority bytes + mutable guard state for one preparation pass.

    Logs and unrelated notes are not authority. They may change freely during
    preparation and are incorporated into the fresh snapshot under the lock.
    The final namespace manifest separately protects every staged hardlink.
    """
    bo_guard.refresh_schedule_scan()
    proposed_builds = tuple(builds)
    root = config.VAULT_PATH.resolve()
    namespace = root / NAMESPACE
    paths = {root / '_schema_rules.json'}
    proposed_ids = {b['build_id'] for b in proposed_builds
                    if isinstance(b, dict) and isinstance(b.get('build_id'), str)}
    guard_ids = set(proposed_ids)
    guard_schedules = {schedule_path} if schedule_path else set()
    graph_entries = list(proposed_builds)
    schedule_bytes = {}
    for name in ('specs', 'schedules', 'templates', 'rules'):
        folder = namespace / name
        paths.add(folder)
        paths.update(folder.rglob('*'))
    # Noncanonical historical spec paths remain read dependencies too.
    for schedule in (namespace / 'schedules').rglob('*'):
        if not schedule.is_file():
            continue
        relative = str(schedule.relative_to(root))
        def read_schedule():
            raw = schedule.read_bytes()
            schedule_bytes[schedule] = raw
            # The index parses text but keeps these same bytes for both the
            # optimistic schedule revision and the approval fingerprint.
            return raw
        _, entries = bo_guard.schedule_document(relative, reader=read_schedule)
        entries = entries or []
        relevant = relative == schedule_path or any(
            isinstance(entry, dict) and entry.get('id') in proposed_ids for entry in entries)
        if relevant:
            guard_schedules.add(relative)
            graph_entries.extend(entries)
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get('spec_path'), str):
                spec = vault.resolve_vault_path(entry['spec_path'])
                paths.add(spec)
                if relevant and spec.is_file():
                    graph_entries.append(frontmatter.loads(spec.read_text()).metadata)
    for entry in graph_entries:
        if not isinstance(entry, dict):
            continue
        for key in ('id', 'build_id', 'goal_root_build_id'):
            if isinstance(entry.get(key), str):
                guard_ids.add(entry[key])
        for key in ('depends_on', 'sequence_after'):
            value = entry.get(key) or []
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        guard_ids.update((item, item.strip()))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError('authoring input contains a symlink')
            digest.update(str((info.st_ino, info.st_mode)).encode())
            if path.is_file():
                digest.update(schedule_bytes[path] if path in schedule_bytes else path.read_bytes())
        except FileNotFoundError:
            digest.update(b'absent')
    from . import write_contract
    signature = (digest.hexdigest(), _guard_inputs(guard_ids, guard_schedules),
            bo_contract.resolve_authoring_contract_path(), config.BO_AUTHORING_CONTRACT_PYTHON,
            os.environ.get('BO_VAULT_ROOT'), bo_guard.get_mode(),
            vault._write_contract_mode(), config.MAX_CONTENT_SIZE,
            tuple(write_contract._CONTENT_REGISTRY.items()),
            frozenset(write_contract._ENFORCED_RULES))
    return InputBinding(proposed_builds, schedule_path, signature)


def refresh(files: dict[str, str]) -> list[str]:
    """Index callbacks follow durable publication and global-lock release."""
    warnings = []
    for hook in _commit_hooks:
        try:
            hook(list(files))
        except Exception:
            logger.exception('authoring committed; index refresh failed')
            warnings.append('authoring committed; index refresh needs retry')
    return warnings


def commit(files: dict[str, str], *, schedule_path: str, expected_revision: str,
           tool: str, result: dict, receipt: tuple[str, str] | None = None,
           dry_run: bool = False, inputs=None) -> list[str]:
    """Validate the exact staged bytes and atomically publish, or write nothing.

    Expensive validation runs outside the global lock. Every result is bound
    to exact authority bytes and mutable guard inputs, checked again under
    the existing global lock and immediately before publication. Changed
    inputs discard preparation; callers retry the full pass outside the lock.
    Dry runs exercise the same checks without staging, receipts or a lock.
    """
    inputs = capture_inputs() if inputs is None else inputs
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
    if inputs.current() != inputs:
        raise PreparationChanged('authoring inputs changed during preparation')
    if dry_run:
        return []
    with vault_lock.registry_lock():
        if inputs.current() != inputs:
            raise PreparationChanged('authoring inputs changed before staging')
        return _publish(root, files, originals, tool, result, receipt, inputs)


def _publish(root, files, originals, tool, result, receipt, inputs):
    # All staging, final checks, exchange, durability and receipt publication
    # remain excluded against both legacy and new cooperating path writers.
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
        if inputs.current() != inputs:
            raise PreparationChanged('authoring inputs changed during staging')
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
    return warnings
