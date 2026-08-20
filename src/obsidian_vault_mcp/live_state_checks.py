"""live_state_checks.py -- deterministic live-reality checks for specific
canonical-state components (build:
vault-canonical-state-live-reality-remediation-v1).

Problem this closes: `check_canonical_state_alignment()` (contradiction_lint.py,
build: vault-infrastructure-state-migration) verifies referential integrity
(does a referenced component_id resolve?) and `verified_at` age, but never
compares a record's *content* against live reality. That structural gap is
why `repo-checkouts-obsidian-web-mcp` asserted a dev-checkout branch that had
silently drifted two days earlier and passed every existing check --
resolved fine, referenced fine, nowhere near the staleness window
(opus-review-phase3-canonical-state-v4, BLOCKER 2).

This module is a narrow, explicit registry: one deterministic Python checker
per `component_id` that needs live-reality verification, keyed by
component_id -- never by a command, path, or branch name read from the
record's own frontmatter/body. The whole point is to check the record's
claims against ground truth, so the record cannot also be the source of what
ground truth should be; expectations live as fixed constants in this
reviewed module. No shell command is ever built from vault content, and no
LLM judgment is involved.

Report-only, same role as `canonical_state_scan.py` and
`check_canonical_state_alignment()`: never writes, never auto-fixes. A
checker exception, an inspection step that cannot complete, or a duplicate
current record for a registered component_id are all findings -- never a
silent pass. Deliberately bounded to the one component this build's causal
gap concerns; adding another component_id requires a reviewed code change to
this file, not a vault edit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from .canonical_state import (
    CanonicalStateRecord,
    DuplicateAuthorityError,
    load_all_records,
    resolve_current_state,
)

# Fixed, reviewed-in-source expectations for repo-checkouts-obsidian-web-mcp.
# See BS 2nd Brain/Alcove/Infrastructure/Canonical State/records/
# repo-checkouts-obsidian-web-mcp.md for the human-readable description this
# mirrors -- kept here as code, not read from there, because that record is
# exactly what this checker exists to verify.
REPO_CHECKOUTS_EXPECTED: dict[str, dict] = {
    "dev": {
        "path": Path("/home/ben_sum/obsidian-web-mcp"),
        "branch": "main",
        "required_paths": (
            "scripts/dreaming.py",
            "scripts/job_miss_check.py",
            "scripts/hot-md-curate.py",
            "scripts/contradiction_lint.py",
            "scripts/canonical_state_scan.py",
            "evals/run_eval.py",
        ),
    },
    "live": {
        "path": Path("/mnt/c/Users/Ben Sum/obsidian-web-mcp"),
        "branch": "feature/vault-tools-v2",
        "required_paths": (),
    },
}


class LiveStateInspectionError(Exception):
    """Raised when a live-state checker cannot complete an inspection step
    (e.g. the git invocation itself failed). Callers MUST surface this as a
    finding, never treat it as equivalent to a clean result."""


def _git_branch(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise LiveStateInspectionError(
            f"git -C {path} rev-parse --abbrev-ref HEAD failed "
            f"(exit {result.returncode}): {result.stderr.strip()[:300]}"
        )
    return result.stdout.strip()


def check_repo_checkouts_obsidian_web_mcp(
    record: CanonicalStateRecord,
    *,
    expected: dict[str, dict] | None = None,
) -> list[dict]:
    """Live-reality checker for component_id `repo-checkouts-obsidian-web-mcp`.

    Compares each expected checkout's live git branch and required
    maintenance-file presence against fixed, reviewed expectations --
    never against anything parsed from ``record`` itself. ``expected``
    defaults to :data:`REPO_CHECKOUTS_EXPECTED`; tests inject a fixture
    mapping so this never has to touch real filesystem paths. ``record`` is
    accepted for interface symmetry with the registry and so a finding can
    cite the record it concerns.
    """
    expected = REPO_CHECKOUTS_EXPECTED if expected is None else expected
    findings: list[dict] = []
    for label, spec in expected.items():
        path: Path = spec["path"]
        if not path.is_dir():
            findings.append(
                {
                    "component_id": record.component_id,
                    "checkout": label,
                    "issue": "path_missing",
                    "detail": f"expected checkout path {path} does not exist",
                }
            )
            continue
        try:
            branch = _git_branch(path)
        except LiveStateInspectionError as exc:
            findings.append(
                {
                    "component_id": record.component_id,
                    "checkout": label,
                    "issue": "inspection_error",
                    "detail": str(exc),
                }
            )
            continue
        if branch != spec["branch"]:
            findings.append(
                {
                    "component_id": record.component_id,
                    "checkout": label,
                    "issue": "branch_mismatch",
                    "detail": f"expected branch {spec['branch']!r}, live branch is {branch!r}",
                }
            )
        for rel in spec["required_paths"]:
            if not (path / rel).exists():
                findings.append(
                    {
                        "component_id": record.component_id,
                        "checkout": label,
                        "issue": "required_path_missing",
                        "detail": f"{path / rel} does not exist",
                    }
                )
    return findings


# component_id -> checker. Closed and explicit by design (spec: "do not
# broaden to every infrastructure component in this build") -- adding a
# component here requires a reviewed code change, never a vault edit.
LIVE_STATE_CHECKERS: dict[str, Callable[[CanonicalStateRecord], list[dict]]] = {
    "repo-checkouts-obsidian-web-mcp": check_repo_checkouts_obsidian_web_mcp,
}


def check_live_state_alignment(
    records_dir: Path,
    *,
    checkers: dict[str, Callable[[CanonicalStateRecord], list[dict]]] | None = None,
) -> list[dict]:
    """Run every registered live-state checker against its current record.

    Report-only, zero-LLM, no arbitrary command execution: each checker is a
    fixed Python function keyed by component_id, not a string sourced from
    vault content. A registered component_id with duplicate current authority
    is a finding; a checker that raises is a finding (fail-closed); a
    registered component_id with no current record on disk produces no
    finding here (that absence is already the existing referential check's
    job, via `check_canonical_state_alignment`).
    """
    checkers = LIVE_STATE_CHECKERS if checkers is None else checkers
    findings: list[dict] = []
    records, _errors = load_all_records(records_dir)
    known_ids = {r.component_id for r in records}
    for component_id, checker in checkers.items():
        if component_id not in known_ids:
            continue
        try:
            record = resolve_current_state(component_id, records_dir)
        except DuplicateAuthorityError as exc:
            findings.append(
                {
                    "component_id": component_id,
                    "issue": "duplicate_authority",
                    "detail": str(exc),
                }
            )
            continue
        if record is None:
            continue
        try:
            findings.extend(checker(record))
        except Exception as exc:  # fail-closed: an inspection failure IS a finding
            findings.append(
                {
                    "component_id": component_id,
                    "issue": "checker_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
    return findings
