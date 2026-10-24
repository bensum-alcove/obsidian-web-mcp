"""Durable, bounded authoring jobs and payload-bound request replay.

Only non-secret authoring inputs and results belong here. A restarted server
never resumes an in-flight operation or invents success: its job is unknown.
A committed synchronous request can recover its atomically published receipt.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import uuid

from . import config

logger = logging.getLogger(__name__)
_INSTANCE = uuid.uuid4().hex
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bo-authoring")
_capacity = threading.BoundedSemaphore(32)
_request_mutex = threading.RLock()
_active_jobs: set[str] = set()


def _root() -> Path:
    # Vault-specific, durable, hidden from vault tools. No runtime credentials.
    return config.VAULT_PATH / ".bo-authoring-state"


def payload_hash(tool: str, payload: dict) -> str:
    return hashlib.sha256(json.dumps({"tool": tool, "payload": payload},
                                    sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _write(path: Path, record: dict):
    from .authoring_transaction import _stage_file, _sync_dir
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _stage_file(path, json.dumps(record, ensure_ascii=False).encode(), 0o600)
    # Persist creation of the metadata directories as well as the file.
    for directory in (path.parent, path.parent.parent, config.VAULT_PATH):
        _sync_dir(directory)


@contextlib.contextmanager
def _locked():
    # One short metadata lock; never held while executing a job or waiting.
    with _request_mutex:
        root = _root()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (root / "requests.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield root
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def _unknown(record: dict) -> dict:
    return {"ok": False, "activated": False, "code": "authoring_job_unknown",
            "job_id": record["job_id"], "status": "unknown",
            "error": "no durable terminal result is available; inspect the schedule before further authoring"}


def bo_authoring_job(job_id: str) -> str:
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        return json.dumps({"ok": False, "status": "unknown", "code": "unknown_job"})
    record = _read(_root() / "jobs" / f"{job_id}.json")
    if record is None:
        return json.dumps({"ok": False, "job_id": job_id, "status": "unknown", "code": "unknown_job"})
    if record["status"] in ("succeeded", "failed"):
        return json.dumps({"ok": record["result"].get("ok", False), "job_id": job_id,
                           "status": record["status"], "result": record["result"]})
    if record["instance"] != _INSTANCE or job_id not in _active_jobs:
        return json.dumps(_unknown(record))
    return json.dumps({"ok": True, "job_id": job_id, "status": record["status"]})


def submit(tool: str, payload: dict, operation, *, request_id: str | None = None,
           wait: bool = True, dry_run: bool = False) -> str:
    """operation(receipt) returns tool JSON; receipt is committed with artifacts."""
    if type(wait) is not bool or type(dry_run) is not bool:
        return json.dumps({"ok": False, "error": "wait and dry_run must be booleans"})
    if request_id is not None and (not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 200):
        return json.dumps({"ok": False, "error": "request_id must contain 1–200 characters"})
    # A preview has zero filesystem side effects, including no job/receipt/cache.
    if dry_run:
        if not wait:
            return json.dumps({"ok": False, "error": "dry_run requires wait=true; previews write no job state"})
        try:
            return operation(None)
        except Exception as exc:
            return json.dumps({"ok": False, "activated": False, "error": str(exc)})
    digest = payload_hash(tool, payload)
    key = hashlib.sha256(request_id.encode()).hexdigest() if request_id else None
    with _locked() as root:
        pointer = _read(root / "requests" / f"{key}.json") if key else None
        if pointer:
            if pointer["payload_hash"] != digest:
                return json.dumps({"ok": False, "activated": False, "code": "request_id_payload_mismatch"})
            record = _read(root / "jobs" / f"{pointer['job_id']}.json")
            if record is None:
                return json.dumps({"ok": False, "code": "authoring_job_unknown", "status": "unknown"})
            if record["status"] in ("succeeded", "failed"):
                return json.dumps(record["result"])
            if record["instance"] != _INSTANCE or record["job_id"] not in _active_jobs:
                if record.get("wait"):
                    from .authoring_transaction import read_receipt
                    recovered = read_receipt(key, digest)
                    if recovered is not None:
                        record.update(status="succeeded", result=recovered)
                        _write(root / "jobs" / f"{record['job_id']}.json", record)
                        return json.dumps(recovered)
                return json.dumps(_unknown(record))
            return json.dumps({"ok": True, "job_id": record["job_id"], "status": record["status"]})
        if not _capacity.acquire(blocking=False):
            return json.dumps({"ok": False, "code": "authoring_queue_full"})
        job_id = uuid.uuid4().hex
        record = {"job_id": job_id, "instance": _INSTANCE, "status": "queued",
                  "payload_hash": digest, "wait": wait}
        job_path = root / "jobs" / f"{job_id}.json"
        try:
            _write(job_path, record)
            if key:
                _write(root / "requests" / f"{key}.json", {"job_id": job_id, "payload_hash": digest})
        except BaseException:
            _capacity.release()
            raise
        _active_jobs.add(job_id)

    def run():
        try:
            record["status"] = "running"
            _write(job_path, record)
            try:
                result = json.loads(operation((key, digest) if key else None))
            except Exception as exc:
                logger.exception("authoring job failed")
                result = {"ok": False, "activated": False, "error": str(exc)}
            record.update(status="succeeded" if result.get("ok") else "failed", result=result)
            try:
                _write(job_path, record)
            except OSError:
                # Publication already has its own durable commit point. Never
                # report a committed graph as rolled back because the derived
                # job-status cache is unavailable. Async polling stays unknown;
                # synchronous retries can recover the transaction receipt.
                logger.exception("authoring terminal job record could not be persisted")
            return json.dumps(result)
        finally:
            with _request_mutex:
                _active_jobs.discard(job_id)
            _capacity.release()
    if wait:
        return run()
    try:
        _pool.submit(run)
    except BaseException:
        with _request_mutex:
            _active_jobs.discard(job_id)
        _capacity.release()
        record.update(status="failed", result={"ok": False, "code": "authoring_worker_unavailable"})
        _write(job_path, record)
        raise
    return json.dumps({"ok": True, "job_id": job_id, "status": "queued"})
