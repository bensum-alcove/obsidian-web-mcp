#!/usr/bin/env python3
"""vault_functional_canary.py — functional (not just process/HTTP) health probe
for a vault's read/query/write/patch/cleanup round trip.

Cron: intended to run every 15 minutes per vault via VAULT_PATH (same
convention as vault-backup.sh's cadence, and the same env-var-selects-vault
convention dreaming.py/job_miss_check.py already use). check-vault-mcp.sh
already proves the MCP process answers HTTP; that is necessary but not
sufficient evidence the Brain can actually read, retrieve, mutate scratch
content and recover indexes -- this script closes that specific gap.

Sequence (vault-observability-slo build spec):
    read known immutable canary
      -> exact (frontmatter) query
      -> hybrid (frontmatter + full-text) query
      -> create scratch note
      -> patch it under an optimistic-concurrency (expected-revision) guard
      -> verify the frontmatter-parse layer sees the patched value
      -> delete the scratch note
      -> verify cleanup

Every step's outcome is reported independently as a LayerResult, never
collapsed into one red/green (see slo.py's "layer" grouping and the spec's
"Layer-specific failure reporting; do not collapse everything into
red/green"). The layer list has a FIXED length and order every run --
downstream steps are marked skipped (not omitted) if a prerequisite failed,
so the output schema never varies with which layer broke.

All writes are confined to config.SCRATCH_DIR_NAME ("_scratch/canary/"),
which vault_search/vault_semantic_search exclude from ordinary retrieval
(config.RETRIEVAL_EXCLUDED_DIRS) but which the frontmatter-parse layer and
vault_list still see -- see config.py's docstring. assert_scratch_scope()
below is a hard safety net: it fails the run (not just warns) if any layer
somehow touched a path outside that namespace.

Architecture note: like dreaming.py/contradiction_lint.py/hot-md-curate.py,
this operates on VAULT_PATH's on-disk content directly via
obsidian_vault_mcp.vault/vault_lock, the same primitives the live MCP server
uses for the same files -- it does not speak MCP-over-HTTP to the live
server process. That means it does NOT validate the live server's own
in-process FrontmatterIndex instance, auth stack, or HTTP transport in
isolation (check-vault-mcp.sh already covers process/HTTP liveness); it
validates the shared on-disk read/write/index-parse layer underneath it,
which is the layer HTTP 200 cannot see through.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import frontmatter

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obsidian_vault_mcp import config  # noqa: E402
from obsidian_vault_mcp import vault_lock  # noqa: E402
from obsidian_vault_mcp.tools.search import vault_search as _vault_search  # noqa: E402

DEFAULT_STATUS_DIR = Path.home() / ".local" / "state" / "vault-observability"

CANARY_DIR = f"{config.SCRATCH_DIR_NAME}/canary"
IMMUTABLE_CANARY_REL = f"{CANARY_DIR}/immutable-canary.md"
IMMUTABLE_SENTINEL = "vault-functional-canary-immutable-fixture-do-not-edit"
IMMUTABLE_CONTENT = (
    "---\n"
    "type: canary-fixture\n"
    "canary_id: immutable\n"
    "managed_by: vault_functional_canary.py\n"
    "---\n\n"
    f"{IMMUTABLE_SENTINEL}\n"
)

LAYER_ORDER = [
    "read_immutable_canary",
    "exact_query",
    "hybrid_query",
    "create_scratch",
    "patch_with_expected_revision",
    "verify_index_sees_patch",
    "cleanup_scratch",
    "verify_cleanup",
    "scratch_scope_guard",
]


class ConcurrentModificationError(RuntimeError):
    """Raised when a scratch file's content changed between read and patch."""


@dataclass
class LayerResult:
    layer: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_locked(resolved_path: Path, data: bytes) -> None:
    """Write inside an already-held vault_lock.path_lock. Mirrors dreaming.py's
    _replace_locked (same tempfile+fsync+os.replace sequence); kept local
    rather than extracted into vault_lock.py to avoid touching that
    already-tested shared module for this build."""
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


def _skipped(layer: str, reason: str) -> LayerResult:
    return LayerResult(layer, False, f"skipped: {reason}")


def ensure_immutable_canary(vault_path: Path) -> LayerResult:
    """Read the immutable canary fixture, creating it on first run only."""
    resolved = vault_path / IMMUTABLE_CANARY_REL
    try:
        if not resolved.exists():
            with vault_lock.path_lock(resolved):
                if not resolved.exists():  # re-check inside the lock
                    _write_locked(resolved, IMMUTABLE_CONTENT.encode("utf-8"))
            return LayerResult("read_immutable_canary", True, "created on first run")
        raw = resolved.read_bytes()
        if IMMUTABLE_SENTINEL.encode("utf-8") not in raw:
            return LayerResult(
                "read_immutable_canary", False,
                "immutable canary fixture content drifted -- expected sentinel not found",
            )
        return LayerResult("read_immutable_canary", True)
    except Exception as exc:
        return LayerResult("read_immutable_canary", False, f"{type(exc).__name__}: {exc}")


def exact_query(vault_path: Path) -> LayerResult:
    """Structured (frontmatter) query -- the same parse primitive the
    production FrontmatterIndex uses (python-frontmatter's frontmatter.load),
    scoped to the canary fixture only (no full-vault rescan; see module
    docstring's architecture note)."""
    resolved = vault_path / IMMUTABLE_CANARY_REL
    try:
        post = frontmatter.load(str(resolved))
        if post.metadata.get("canary_id") != "immutable":
            return LayerResult("exact_query", False, "frontmatter field canary_id not found/mismatched")
        return LayerResult("exact_query", True)
    except Exception as exc:
        return LayerResult("exact_query", False, f"{type(exc).__name__}: {exc}")


def hybrid_query(vault_path: Path) -> LayerResult:
    """Full-text query via the production vault_search tool (ripgrep-backed,
    Python fallback otherwise) -- combined with exact_query's structural
    check, this exercises both retrieval paths a real client uses."""
    try:
        raw = _vault_search(
            query=IMMUTABLE_SENTINEL,
            path_prefix=CANARY_DIR,
            file_pattern="*.md",
            max_results=5,
            context_lines=0,
        )
        payload = json.loads(raw)
        if payload.get("error"):
            return LayerResult("hybrid_query", False, str(payload["error"]))
        hits = [m for m in payload.get("results", []) if m["path"].endswith("immutable-canary.md")]
        if not hits:
            return LayerResult("hybrid_query", False, "text search did not find the immutable canary fixture")
        return LayerResult("hybrid_query", True)
    except Exception as exc:
        return LayerResult("hybrid_query", False, f"{type(exc).__name__}: {exc}")


def create_scratch(vault_path: Path, run_id: str) -> tuple[LayerResult, str | None, str | None]:
    """Create a fresh, uniquely-named scratch note. Returns (result, rel_path, revision)."""
    rel_path = f"{CANARY_DIR}/probe-{run_id}.md"
    resolved = vault_path / rel_path
    content = (
        "---\n"
        "type: canary-fixture\n"
        f"canary_id: {run_id}\n"
        "phase: created\n"
        "---\n\n"
        f"scratch probe {run_id}\n"
    )
    try:
        with vault_lock.path_lock(resolved):
            if resolved.exists():
                return LayerResult("create_scratch", False, "probe path already existed"), None, None
            _write_locked(resolved, content.encode("utf-8"))
        revision = _sha256_hex(resolved.read_bytes())
        return LayerResult("create_scratch", True), rel_path, revision
    except Exception as exc:
        return LayerResult("create_scratch", False, f"{type(exc).__name__}: {exc}"), None, None


def patch_with_expected_revision(
    vault_path: Path, rel_path: str, expected_revision: str, run_id: str
) -> LayerResult:
    """Optimistic-concurrency guarded patch. Mirrors dreaming.py's
    scan_hash/apply_autofix hash-guard pattern (dreaming-safe-remediation-v2):
    re-read and re-hash the file INSIDE the same path_lock acquisition that
    guards the write; refuse to write if the hash no longer matches
    expected_revision (a concurrent writer landed since create_scratch)."""
    resolved = vault_path / rel_path
    new_content = (
        "---\n"
        "type: canary-fixture\n"
        f"canary_id: {run_id}\n"
        "phase: patched\n"
        "---\n\n"
        f"scratch probe {run_id} -- patched\n"
    )
    try:
        with vault_lock.path_lock(resolved):
            current = resolved.read_bytes()
            if _sha256_hex(current) != expected_revision:
                raise ConcurrentModificationError(
                    f"{rel_path} changed since create_scratch; refusing to overwrite"
                )
            _write_locked(resolved, new_content.encode("utf-8"))
        return LayerResult("patch_with_expected_revision", True)
    except Exception as exc:
        return LayerResult("patch_with_expected_revision", False, f"{type(exc).__name__}: {exc}")


def verify_index_sees_patch(vault_path: Path, rel_path: str) -> LayerResult:
    resolved = vault_path / rel_path
    try:
        post = frontmatter.load(str(resolved))
        if post.metadata.get("phase") != "patched":
            return LayerResult(
                "verify_index_sees_patch", False,
                "parsed frontmatter does not reflect the patched phase",
            )
        return LayerResult("verify_index_sees_patch", True)
    except Exception as exc:
        return LayerResult("verify_index_sees_patch", False, f"{type(exc).__name__}: {exc}")


def cleanup_scratch(vault_path: Path, rel_path: str) -> LayerResult:
    resolved = vault_path / rel_path
    try:
        with vault_lock.path_lock(resolved):
            resolved.unlink()
        return LayerResult("cleanup_scratch", True)
    except Exception as exc:
        return LayerResult("cleanup_scratch", False, f"{type(exc).__name__}: {exc}")


def verify_cleanup(vault_path: Path, rel_path: str) -> LayerResult:
    resolved = vault_path / rel_path
    if resolved.exists():
        return LayerResult("verify_cleanup", False, "scratch probe still present after cleanup")
    return LayerResult("verify_cleanup", True)


def assert_scratch_scope(touched: list[str]) -> LayerResult:
    """Hard safety net for the completion criterion 'canary never mutates
    non-scratch knowledge': every path this run wrote/deleted must sit under
    CANARY_DIR. Fails the run (not just warns) if violated."""
    offenders = [p for p in touched if not p.startswith(CANARY_DIR + "/") and p != IMMUTABLE_CANARY_REL]
    if offenders:
        return LayerResult("scratch_scope_guard", False, f"touched paths outside scratch: {offenders}")
    return LayerResult("scratch_scope_guard", True)


def run_canary(vault_path: Path, now: datetime) -> dict:
    run_id = now.strftime("%Y%m%dT%H%M%S%f")
    results: dict[str, LayerResult] = {}
    touched: list[str] = [IMMUTABLE_CANARY_REL]

    results["read_immutable_canary"] = ensure_immutable_canary(vault_path)

    if results["read_immutable_canary"].ok:
        results["exact_query"] = exact_query(vault_path)
        results["hybrid_query"] = hybrid_query(vault_path)
    else:
        results["exact_query"] = _skipped("exact_query", "read_immutable_canary failed")
        results["hybrid_query"] = _skipped("hybrid_query", "read_immutable_canary failed")

    create_result, rel_path, revision = create_scratch(vault_path, run_id)
    results["create_scratch"] = create_result
    if rel_path is not None:
        touched.append(rel_path)

    if create_result.ok:
        results["patch_with_expected_revision"] = patch_with_expected_revision(
            vault_path, rel_path, revision, run_id
        )
    else:
        results["patch_with_expected_revision"] = _skipped(
            "patch_with_expected_revision", "create_scratch failed"
        )

    if results["patch_with_expected_revision"].ok:
        results["verify_index_sees_patch"] = verify_index_sees_patch(vault_path, rel_path)
    else:
        results["verify_index_sees_patch"] = _skipped(
            "verify_index_sees_patch", "patch_with_expected_revision failed"
        )

    if rel_path is not None and (vault_path / rel_path).exists():
        results["cleanup_scratch"] = cleanup_scratch(vault_path, rel_path)
    else:
        results["cleanup_scratch"] = _skipped("cleanup_scratch", "no scratch probe to clean up")

    if rel_path is not None:
        results["verify_cleanup"] = verify_cleanup(vault_path, rel_path)
    else:
        results["verify_cleanup"] = _skipped("verify_cleanup", "create_scratch failed")

    results["scratch_scope_guard"] = assert_scratch_scope(touched)

    layers = [results[name].to_dict() for name in LAYER_ORDER]
    failing = [layer["layer"] for layer in layers if not layer["ok"]]

    return {
        "vault_path": str(vault_path),
        "run_id": run_id,
        "checked_at": now.isoformat(),
        "overall_ok": not failing,
        "layers_failing": failing,
        "layers": layers,
    }


def write_status(status: dict, vault_name: str, status_dir: Path) -> Path:
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / f"canary-{vault_name}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault-path", type=Path, default=config.VAULT_PATH)
    parser.add_argument("--vault-name", default=os.environ.get("VAULT_NAME", config.VAULT_PATH.name))
    parser.add_argument("--status-dir", type=Path, default=DEFAULT_STATUS_DIR)
    args = parser.parse_args()

    # hybrid_query() calls the shared vault_search tool, which resolves
    # paths against the config.VAULT_PATH global rather than an explicit
    # parameter -- keep it in sync with --vault-path so an override doesn't
    # silently diverge from what the rest of this script operates on.
    config.VAULT_PATH = args.vault_path

    now = datetime.now(timezone.utc)
    status = run_canary(args.vault_path, now)
    status["vault_name"] = args.vault_name
    out_path = write_status(status, args.vault_name, args.status_dir)

    print(json.dumps(status, indent=2))
    print(f"status written to {out_path}", file=sys.stderr)
    return 0 if status["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
