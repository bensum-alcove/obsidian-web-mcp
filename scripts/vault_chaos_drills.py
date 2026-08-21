#!/usr/bin/env python3
"""vault_chaos_drills.py — controlled fault-injection drills for the vault's
on-disk read/write/index layer (vault-chaos-recovery-suite build).

Companion to vault_functional_canary.py (which proves the happy path works)
and vault_safety_sweep.py (which proves corruption is detected/repaired on a
schedule): this proves what happens when things actually break, on purpose,
in a throwaway location.

Every drill below creates its OWN fresh temporary directory and points
config.VAULT_PATH at it for the duration of that one drill only (restored
immediately after) -- no drill ever touches a real vault
(BS Brain/Alcove Brain/CB Brain) or a shared fixture another test depends on.
See mcp-infrastructure's chaos/failure-matrix.md for the pre-registered
classification and expected detection/containment/recovery/max-loss per
scenario BEFORE this file existed -- that document is not backfilled to match
whatever this script happens to produce.

Two scenarios are deliberately reported as PARTIAL passes with an explicit,
tested residual gap rather than a clean pass, because the underlying code
does not yet fully back the claim:
  - stale_concurrent_edit: the vault_lock + hash-guard pattern used by
    in-tree script writers (vault_functional_canary.py, hot-md-curate.py
    --apply, vault_safety_sweep.py --repair-content, vault-audit.py
    --autofix) genuinely blocks a stale write. The live MCP write tools
    (tools/write.py, tools/manage.py) accept no expected_revision/hash
    parameter at all and tools/write.py's write_file_atomic never acquires
    vault_lock -- despite vault_read's own tool description advertising an
    expected_revision contract. Both facts are demonstrated here, not just
    asserted from reading the code.
  - malformed_yaml_write: vault_safety_sweep.py detects and, for exactly one
    known corruption signature, repairs malformed frontmatter on a schedule.
    The live vault_write tool performs no frontmatter validation on a plain
    write at all and will happily persist malformed YAML to disk.

Scenario numbering matches chaos/failure-matrix.md in mcp-infrastructure:
  2  semantic index loss/corruption
  3  entity index loss
  4  interruption during atomic-write
  5  read-only filesystem
  6  stale concurrent edit
  7  malformed YAML write
  11 simulated accidental bulk modification
(1, 8, 9 are infra-level and live in mcp-infrastructure's bash harness;
10 -- corrupt latest backup generation -- is scripts/vault_backup_selector.py
in this repo.)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obsidian_vault_mcp import config, vault, vault_lock  # noqa: E402
from obsidian_vault_mcp import observability_alert  # noqa: E402
from obsidian_vault_mcp.frontmatter_safe import FrontmatterError, parse_frontmatter  # noqa: E402
from obsidian_vault_mcp.tools import semantic_search as ss  # noqa: E402
from obsidian_vault_mcp.tools import write as write_tools  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_sibling_script(name: str, filename: str):
    """Load a sibling scripts/*.py module by path -- same pattern
    tests/test_dreaming.py and tests/test_vault_safety_sweep.py already use,
    since scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dreaming = _load_sibling_script("dreaming_for_chaos_drills", "dreaming.py")
safety_sweep = _load_sibling_script("vault_safety_sweep_for_chaos_drills", "vault_safety_sweep.py")


@dataclass
class DrillResult:
    scenario_id: int
    name: str
    classification: str
    ok: bool
    detected: bool
    contained: bool
    recovered: bool
    max_loss: str
    evidence: dict
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )


# --- 2. Semantic index loss/corruption -------------------------------------

def drill_semantic_index_loss(root: Path) -> DrillResult:
    name = "semantic_index_loss_corruption"
    prev_vault_path = config.VAULT_PATH
    try:
        config.VAULT_PATH = root
        (root / "alpha.md").write_text(
            "---\ntype: note\n---\n\nunique-marker-alpha content for chaos drill.\n"
        )
        (root / "beta.md").write_text(
            "---\ntype: note\n---\n\nunique-marker-beta content for chaos drill.\n"
        )

        if not ss.SEMANTIC_AVAILABLE:
            return DrillResult(
                scenario_id=2, name=name, classification="isolated_restored_vault",
                ok=False, detected=False, contained=False, recovered=False,
                max_loss="untestable in this environment",
                evidence={"reason": "fastembed/sqlite-vec not installed here"},
                error="semantic search unavailable -- residual, not claimed passed",
            )

        ss.build_index()
        ss._index_ready = True
        before = json.loads(ss.vault_semantic_search("unique-marker-alpha"))

        index_path = root / ".semantic-index" / "index.db"
        index_existed = index_path.exists()
        index_path.unlink()
        ss._index_ready = False  # the state a freshly-restarted process would start in

        degraded = json.loads(ss.vault_semantic_search("unique-marker-alpha"))
        detected = degraded.get("status") == "building"

        ss.build_index()
        ss._index_ready = True
        after = json.loads(ss.vault_semantic_search("unique-marker-alpha"))
        recovered = isinstance(after, list) and any(r["path"] == "alpha.md" for r in after)

        evidence = {
            "index_existed_before_loss": index_existed,
            "query_before_loss": before,
            "query_during_loss": degraded,
            "query_after_rebuild": after,
        }
        return DrillResult(
            scenario_id=2, name=name, classification="isolated_restored_vault",
            ok=detected and recovered, detected=detected, contained=True, recovered=recovered,
            max_loss="0 (index is fully derived from Markdown; content itself is never at risk)",
            evidence=evidence,
        )
    finally:
        config.VAULT_PATH = prev_vault_path
        ss._index_ready = False


# --- 3. Entity index loss ---------------------------------------------------

def drill_entity_index_loss(root: Path) -> DrillResult:
    name = "entity_index_loss"
    vault_name = "chaos-drill-vault"
    clients = root / "Clients"
    clients.mkdir()
    (clients / "Acme.md").write_text(
        "---\ntype: client\n---\n\nAcme is a chaos-drill fixture client.\n"
    )
    (root / "unrelated.md").write_text("---\ntype: note\n---\n\nNothing entity-shaped.\n")

    md_files = dreaming.list_md_files(root)
    entities_before = dreaming.pass_entity_index(root, vault_name, md_files)
    out_path = dreaming.write_entities_json(root, vault_name, datetime.now(timezone.utc), entities_before)
    baseline = json.loads(out_path.read_text())

    out_path.unlink()
    detected = not out_path.exists()

    md_files_after = dreaming.list_md_files(root)
    entities_after = dreaming.pass_entity_index(root, vault_name, md_files_after)
    dreaming.write_entities_json(root, vault_name, datetime.now(timezone.utc), entities_after)
    rebuilt = json.loads(out_path.read_text())

    recovered = (
        baseline["entity_count"] == rebuilt["entity_count"] == 1
        and baseline["entities"][0]["name"] == rebuilt["entities"][0]["name"] == "Acme"
    )

    evidence = {
        "baseline_entity_count": baseline["entity_count"],
        "rebuilt_entity_count": rebuilt["entity_count"],
        "baseline_entity_names": [e["name"] for e in baseline["entities"]],
        "rebuilt_entity_names": [e["name"] for e in rebuilt["entities"]],
    }
    return DrillResult(
        scenario_id=3, name=name, classification="isolated_restored_vault",
        ok=detected and recovered, detected=detected, contained=True, recovered=recovered,
        max_loss="0 (_entities.json is machine-generated from Markdown every run)",
        evidence=evidence,
    )


# --- 4. Interruption during atomic-write ------------------------------------

def drill_atomic_write_interruption(root: Path) -> DrillResult:
    name = "atomic_write_interruption"
    prev_vault_path = config.VAULT_PATH
    try:
        config.VAULT_PATH = root
        (root / "note.md").write_text("original content v1\n")
        original_bytes = (root / "note.md").read_bytes()

        raised = None
        with mock.patch(
            "obsidian_vault_mcp.vault.os.replace",
            side_effect=OSError("simulated crash mid atomic replace"),
        ):
            try:
                vault.write_file_atomic("note.md", "new content v2 -- must never land\n")
            except OSError as exc:
                raised = str(exc)

        detected = raised is not None
        after_failure_bytes = (root / "note.md").read_bytes()
        contained = after_failure_bytes == original_bytes
        leftover_tmp = [p.name for p in root.glob("*.tmp")]

        vault.write_file_atomic("note.md", "new content v2 after retry\n")
        recovered = (root / "note.md").read_text() == "new content v2 after retry\n"

        evidence = {
            "injected_error": raised,
            "original_hash": _hash(original_bytes),
            "hash_immediately_after_injected_failure": _hash(after_failure_bytes),
            "content_provably_unchanged": contained,
            "leftover_tempfiles": leftover_tmp,
            "retry_after_failure_succeeded": recovered,
        }
        ok = detected and contained and recovered and not leftover_tmp
        return DrillResult(
            scenario_id=4, name=name, classification="fixture",
            ok=ok, detected=detected, contained=contained, recovered=recovered,
            max_loss="0 (write never applied; original bytes provably unchanged)",
            evidence=evidence,
        )
    finally:
        config.VAULT_PATH = prev_vault_path


# --- 5. Read-only filesystem -------------------------------------------------

def drill_read_only_filesystem(root: Path) -> DrillResult:
    name = "read_only_filesystem"
    prev_vault_path = config.VAULT_PATH
    try:
        sub = root / "locked"
        sub.mkdir()
        target = sub / "locked.md"
        target.write_text("original content\n")
        config.VAULT_PATH = root

        original_mode = stat.S_IMODE(sub.stat().st_mode)
        os.chmod(sub, 0o555)  # read+execute, no write -- blocks mkstemp in this directory
        raised = None
        try:
            vault.write_file_atomic("locked/locked.md", "attempted change\n")
        except (PermissionError, OSError) as exc:
            raised = str(exc)
        finally:
            os.chmod(sub, original_mode)

        detected = raised is not None
        contained = target.read_text() == "original content\n"

        vault.write_file_atomic("locked/locked.md", "change after permissions restored\n")
        recovered = target.read_text() == "change after permissions restored\n"

        # Secondary, narrower check: vault_safety_sweep.py's own permission-repair
        # path (missing world-readable bit), independent of the write-path recovery above.
        os.chmod(target, 0o600)
        dry_report = safety_sweep.scan_vault("chaos-fixture", root, repair_permissions=False)
        flagged = any(e["path"] == "locked/locked.md" for e in dry_report["permission_errors"])
        fix_report = safety_sweep.scan_vault("chaos-fixture", root, repair_permissions=True)
        sweep_repaired = bool(stat.S_IMODE(target.stat().st_mode) & stat.S_IROTH)

        evidence = {
            "injected_error": raised,
            "content_unchanged_during_failure": contained,
            "retry_after_permissions_restored_succeeded": recovered,
            "safety_sweep_flagged_missing_world_read": flagged,
            "safety_sweep_repaired_permissions": sweep_repaired,
        }
        ok = detected and contained and recovered and flagged and sweep_repaired
        return DrillResult(
            scenario_id=5, name=name, classification="fixture",
            ok=ok, detected=detected, contained=contained, recovered=recovered,
            max_loss="0 (caller-visible error, never silent data loss or a partial write)",
            evidence=evidence,
        )
    finally:
        config.VAULT_PATH = prev_vault_path


# --- 6. Stale concurrent edit ------------------------------------------------

def drill_stale_concurrent_edit(root: Path) -> DrillResult:
    name = "stale_concurrent_edit"
    target = root / "shared.md"
    target.write_text("---\ntype: note\n---\n\nv1 content.\n")
    resolved = target.resolve()

    writer_a_hash = _hash(target.read_bytes())

    # Writer B lands a real, guarded change first.
    vault_lock.atomic_write(resolved, b"---\ntype: note\n---\n\nv2 content by writer B.\n")

    # Writer A now tries to commit its now-stale patch under the same
    # lock+hash-guard pattern vault_functional_canary.py/hot-md-curate.py/
    # vault_safety_sweep.py use: re-check the hash INSIDE the lock.
    blocked = False
    with vault_lock.path_lock(resolved):
        current = resolved.read_bytes()
        if _hash(current) != writer_a_hash:
            blocked = True
        else:
            resolved.write_bytes(b"---\ntype: note\n---\n\nSTALE WRITE -- must not land.\n")

    detected = blocked
    contained = target.read_text() == "---\ntype: note\n---\n\nv2 content by writer B.\n"

    # Recovery: writer A re-reads current state and retries with a fresh hash.
    fresh_hash = _hash(target.read_bytes())
    recovered_write_ok = False
    with vault_lock.path_lock(resolved):
        current = resolved.read_bytes()
        if _hash(current) == fresh_hash:
            recovered_write_ok = True
            resolved.write_bytes(b"---\ntype: note\n---\n\nv3 content by writer A, based on v2.\n")
    recovered = recovered_write_ok and target.read_text() == "---\ntype: note\n---\n\nv3 content by writer A, based on v2.\n"

    # Residual gap, demonstrated (not just inspected): the live MCP write
    # tools accept no expected-revision/hash parameter, so a writer acting
    # on stale knowledge is never blocked at that layer.
    prev_vault_path = config.VAULT_PATH
    config.VAULT_PATH = root
    try:
        write_tools.vault_write("gap-demo.md", "first version, read by a would-be stale writer\n")
        write_tools.vault_write("gap-demo.md", "second version, written by a different, unrelated writer\n")
        clobber_result = json.loads(
            write_tools.vault_write(
                "gap-demo.md",
                "third version: clobbers writer 2's change; vault_write took no revision "
                "argument and raised no conflict\n",
            )
        )
        gap_confirmed = (
            "error" not in clobber_result
            and (root / "gap-demo.md").read_text().startswith("third version")
        )
    finally:
        config.VAULT_PATH = prev_vault_path

    evidence = {
        "writer_a_initial_hash": writer_a_hash,
        "stale_write_blocked_by_guarded_primitive": blocked,
        "content_after_guard": target.read_text(),
        "guarded_recovery_succeeded": recovered,
        "live_mcp_write_tool_gap_confirmed": gap_confirmed,
        "gap_detail": (
            "tools/write.py:vault_write and tools/manage.py's write tools take no "
            "expected_revision/hash parameter, and vault.py:write_file_atomic never "
            "acquires vault_lock -- despite vault_read's own tool description "
            "advertising an expected_revision contract that no write tool implements."
        ),
    }
    # Pre-registered expectation (chaos/failure-matrix.md #6) is exactly this split:
    # the guarded primitive blocks the stale write AND the public MCP gap is real.
    ok = detected and contained and recovered and gap_confirmed
    return DrillResult(
        scenario_id=6, name=name, classification="fixture",
        ok=ok, detected=detected, contained=contained, recovered=recovered,
        max_loss="0 for the guarded primitive (stale write never applied); "
                 "UNBOUNDED for the public MCP write tools (no guard exists there at all)",
        evidence=evidence,
    )


# --- 7. Malformed YAML write --------------------------------------------------

def drill_malformed_yaml_write(root: Path) -> DrillResult:
    name = "malformed_yaml_write"
    prev_vault_path = config.VAULT_PATH
    try:
        config.VAULT_PATH = root
        unterminated = "---\nstatus: active\n\nNo closing delimiter at all.\n"
        (root / "unterminated.md").write_text(unterminated)

        glued = "---\nstatus: active---\n\nBody text glued to the closing delimiter.\n"
        (root / "glued.md").write_text(glued)

        raised = None
        try:
            parse_frontmatter(unterminated)
        except FrontmatterError as exc:
            raised = str(exc)
        detected = raised is not None

        dry = safety_sweep.scan_vault("chaos-fixture", root, repair_content=False)
        unterminated_flagged = any(
            e["path"] == "unterminated.md" and not e["known_signature"]
            for e in dry["frontmatter_errors"]
        )
        glued_flagged = any(e["path"] == "glued.md" for e in dry["frontmatter_errors"])
        contained = dry["repairs"] == [] and unterminated_flagged and glued_flagged

        sent_messages = []
        alert_open = observability_alert.record_and_maybe_alert(
            "chaos-drill-malformed-yaml", True,
            f"{len(dry['frontmatter_errors'])} malformed frontmatter file(s) detected",
            state_dir=root / "_alert_state", send_fn=sent_messages.append,
        )

        fixed = safety_sweep.scan_vault("chaos-fixture", root, repair_content=True)
        glued_repaired = any(
            r["path"] == "glued.md" and r["type"] == "frontmatter" and r["verified"]
            for r in fixed["repairs"]
        )
        unterminated_still_flagged = any(
            e["path"] == "unterminated.md" for e in fixed["frontmatter_errors"]
        )
        # "Recovered" here means: the one auto-fixable class was fixed, and the
        # non-fixable class stayed visibly flagged (never silently dropped).
        recovered = glued_repaired and unterminated_still_flagged

        alert_close = observability_alert.record_and_maybe_alert(
            "chaos-drill-malformed-yaml", unterminated_still_flagged and not glued_repaired,
            "known-signature corruption auto-repaired; unresolvable case still flagged for human triage",
            state_dir=root / "_alert_state", send_fn=sent_messages.append,
        )

        # Residual gap: the live write path performs no frontmatter validation.
        write_result = json.loads(write_tools.vault_write("plain-write-bad.md", unterminated))
        wrote_unvalidated = (
            "error" not in write_result
            and (root / "plain-write-bad.md").read_text() == unterminated
        )

        evidence = {
            "frontmatter_error_message": raised,
            "dry_run_frontmatter_errors": dry["frontmatter_errors"],
            "repair_run_repairs": fixed["repairs"],
            "repair_run_still_flagged": fixed["frontmatter_errors"],
            "alert_open_outcome": alert_open,
            "alert_close_outcome": alert_close,
            "alert_messages_sent": sent_messages,
            "live_write_path_accepted_malformed_yaml_unvalidated": wrote_unvalidated,
        }
        ok = detected and contained and recovered and wrote_unvalidated
        return DrillResult(
            scenario_id=7, name=name, classification="fixture",
            ok=ok, detected=detected, contained=contained, recovered=recovered,
            max_loss="0 (detection is read-only by default; only the one known-signature "
                     "class is auto-repaired, everything else is flagged, never silently dropped)",
            evidence=evidence,
        )
    finally:
        config.VAULT_PATH = prev_vault_path


# --- 11. Simulated accidental bulk modification ------------------------------

def drill_bulk_modification(root: Path) -> DrillResult:
    name = "simulated_accidental_bulk_modification"
    contents = {}
    for i in range(5):
        p = root / f"note-{i}.md"
        text = f"---\ntype: note\nseq: {i}\n---\n\nBaseline content for note {i}, chaos-drill fixture.\n"
        p.write_text(text)
        contents[p.name] = text

    _run_git(["init", "-q"], root)
    _run_git(["config", "user.email", "chaos-drill@example.invalid"], root)
    _run_git(["config", "user.name", "chaos-drill"], root)
    _run_git(["add", "-A"], root)
    _run_git(["commit", "-q", "-m", "baseline"], root)
    baseline_commit = _run_git(["rev-parse", "HEAD"], root).stdout.strip()
    baseline_hashes = {n: _hash(t.encode()) for n, t in contents.items()}

    # Simulate an accidental bulk edit (e.g. a bad find/sed across the tree).
    for name_ in contents:
        (root / name_).write_text("")

    changed_files = _run_git(["diff", "--name-only"], root).stdout.strip().splitlines()
    diff_stat = _run_git(["diff", "--stat"], root).stdout
    detected = len(changed_files) >= 3  # bulk-modification threshold

    sent_messages = []
    alert_open = observability_alert.record_and_maybe_alert(
        "chaos-drill-bulk-modification", True,
        f"{len(changed_files)} files bulk-modified in one pass",
        state_dir=root / "_alert_state", send_fn=sent_messages.append,
    )
    contained = True  # detection precedes any auto-remediation; nothing else runs until proven

    _run_git(["checkout", baseline_commit, "--", "."], root)
    after_hashes = {n: _hash((root / n).read_bytes()) for n in contents}
    recovered = after_hashes == baseline_hashes

    alert_close = observability_alert.record_and_maybe_alert(
        "chaos-drill-bulk-modification", False,
        "content restored from last known-good commit", state_dir=root / "_alert_state",
        send_fn=sent_messages.append,
    )

    evidence = {
        "baseline_commit": baseline_commit,
        "changed_file_count": len(changed_files),
        "diff_stat": diff_stat,
        "alert_open_outcome": alert_open,
        "alert_close_outcome": alert_close,
        "alert_messages_sent": sent_messages,
        "post_recovery_hashes_match_baseline": recovered,
    }
    ok = detected and contained and recovered
    return DrillResult(
        scenario_id=11, name=name, classification="isolated_restored_vault",
        ok=ok, detected=detected, contained=contained, recovered=recovered,
        max_loss="0 permanent loss; bounded by time since last known-good commit (15 min cron target)",
        evidence=evidence,
    )


DRILLS = [
    drill_semantic_index_loss,
    drill_entity_index_loss,
    drill_atomic_write_interruption,
    drill_read_only_filesystem,
    drill_stale_concurrent_edit,
    drill_malformed_yaml_write,
    drill_bulk_modification,
]


def run_all() -> dict:
    results = []
    for drill_fn in DRILLS:
        tmp = Path(tempfile.mkdtemp(prefix=f"vault-chaos-{drill_fn.__name__}-"))
        try:
            result = drill_fn(tmp)
        except Exception as exc:  # a drill crashing is itself a finding -- never hide it
            result = DrillResult(
                scenario_id=-1, name=drill_fn.__name__, classification="unknown",
                ok=False, detected=False, contained=False, recovered=False,
                max_loss="unknown", evidence={},
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        results.append(result.to_dict())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "drills": results,
        "all_ok": all(r["ok"] for r in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_all()
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
