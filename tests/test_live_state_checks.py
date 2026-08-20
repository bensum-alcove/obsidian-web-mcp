"""Tests for obsidian_vault_mcp.live_state_checks
(build: vault-canonical-state-live-reality-remediation-v1)."""

import subprocess
from pathlib import Path

import pytest

from obsidian_vault_mcp.live_state_checks import (
    LiveStateInspectionError,
    check_live_state_alignment,
    check_repo_checkouts_obsidian_web_mcp,
)
from obsidian_vault_mcp.canonical_state import CanonicalStateRecord


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def _make_repo(root: Path, branch: str, files: tuple[str, ...] = ()) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", branch)
    _git(root, "config", "user.email", "test@test.com")
    _git(root, "config", "user.name", "test")
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
        _git(root, "add", rel)
    _git(root, "commit", "-q", "--allow-empty", "-m", "init")
    return root


def _record(component_id: str = "repo-checkouts-obsidian-web-mcp", source: str = "test") -> CanonicalStateRecord:
    return CanonicalStateRecord(
        path=Path("dummy.md"),
        component_id=component_id,
        state="active",
        content_updated="2026-08-09",
        verified_at="2026-08-20",
        source=source,
        status_changed_at=None,
        superseded_by=None,
        supersedes=None,
    )


REQUIRED = (
    "scripts/dreaming.py",
    "scripts/job_miss_check.py",
    "scripts/hot-md-curate.py",
    "scripts/contradiction_lint.py",
    "scripts/canonical_state_scan.py",
    "evals/run_eval.py",
)


def _expected(tmp_path, dev_branch="main", dev_files=REQUIRED, live_branch="feature/vault-tools-v2"):
    dev = _make_repo(tmp_path / "dev", dev_branch, dev_files)
    live = _make_repo(tmp_path / "live", live_branch)
    return {
        "dev": {"path": dev, "branch": "main", "required_paths": REQUIRED},
        "live": {"path": live, "branch": "feature/vault-tools-v2", "required_paths": ()},
    }


def test_intended_state_passes(tmp_path):
    expected = _expected(tmp_path)
    findings = check_repo_checkouts_obsidian_web_mcp(_record(), expected=expected)
    assert findings == []


def test_flags_wrong_branch_fixture(tmp_path):
    """Reproduces the exact proven failure: dev checkout drifted onto the
    feature branch instead of main."""
    expected = _expected(tmp_path, dev_branch="feature/vault-tools-v2")
    findings = check_repo_checkouts_obsidian_web_mcp(_record(), expected=expected)
    branch_findings = [f for f in findings if f["issue"] == "branch_mismatch" and f["checkout"] == "dev"]
    assert len(branch_findings) == 1
    assert "main" in branch_findings[0]["detail"]
    assert "feature/vault-tools-v2" in branch_findings[0]["detail"]


def test_flags_missing_required_scripts(tmp_path):
    expected = _expected(tmp_path, dev_files=("scripts/dreaming.py",))
    findings = check_repo_checkouts_obsidian_web_mcp(_record(), expected=expected)
    missing = {f["detail"] for f in findings if f["issue"] == "required_path_missing"}
    assert len(missing) == len(REQUIRED) - 1
    assert any("job_miss_check.py" in d for d in missing)
    assert not any("dreaming.py" in d for d in missing)


def test_missing_checkout_path_is_flagged(tmp_path):
    expected = _expected(tmp_path)
    expected["dev"]["path"] = tmp_path / "does-not-exist"
    findings = check_repo_checkouts_obsidian_web_mcp(_record(), expected=expected)
    assert any(f["issue"] == "path_missing" and f["checkout"] == "dev" for f in findings)


def test_inspection_error_does_not_silently_pass(tmp_path):
    """A path that exists but isn't a git repo must surface as a finding,
    never be treated as a pass."""
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    expected = _expected(tmp_path)
    expected["dev"]["path"] = not_a_repo
    findings = check_repo_checkouts_obsidian_web_mcp(_record(), expected=expected)
    assert any(f["issue"] == "inspection_error" and f["checkout"] == "dev" for f in findings)


def test_checker_never_reads_record_source_as_a_command(tmp_path):
    """The whole point of this checker is verifying record content against
    ground truth -- it must never build its checks from record fields, so a
    hostile/malformed `source` value has zero effect on the result."""
    expected = _expected(tmp_path)
    benign = check_repo_checkouts_obsidian_web_mcp(_record(source="benign"), expected=expected)
    hostile = check_repo_checkouts_obsidian_web_mcp(
        _record(source="; rm -rf / #`whoami`$(id)"), expected=expected
    )
    assert benign == hostile == []


def test_git_branch_primitive_raises_on_non_repo(tmp_path):
    """check_repo_checkouts_obsidian_web_mcp catches LiveStateInspectionError
    and turns it into a finding -- this asserts the underlying primitive
    actually raises, so that fail-closed behavior is real error handling,
    not an accidental empty-list default."""
    from obsidian_vault_mcp.live_state_checks import _git_branch

    not_a_repo = tmp_path / "nope"
    not_a_repo.mkdir()
    with pytest.raises(LiveStateInspectionError):
        _git_branch(not_a_repo)


def _write_record(dir_path: Path, name: str, frontmatter: str) -> Path:
    path = dir_path / name
    path.write_text(f"---\n{frontmatter}\n---\n\nbody\n", encoding="utf-8")
    return path


def test_check_live_state_alignment_registry_wiring(tmp_path):
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    expected = _expected(tmp_path)
    _write_record(
        records_dir,
        "repo-checkouts-obsidian-web-mcp.md",
        "type: canonical-state\n"
        "component_id: repo-checkouts-obsidian-web-mcp\n"
        "state: active\n"
        "content_updated: '2026-08-09'\n"
        "verified_at: '2026-08-20'\n"
        "source: 'test'\n",
    )
    checkers = {
        "repo-checkouts-obsidian-web-mcp": lambda record: check_repo_checkouts_obsidian_web_mcp(
            record, expected=expected
        )
    }
    findings = check_live_state_alignment(records_dir, checkers=checkers)
    assert findings == []


def test_check_live_state_alignment_flags_drift_end_to_end(tmp_path):
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    drifted_expected = _expected(tmp_path, dev_branch="feature/vault-tools-v2")
    _write_record(
        records_dir,
        "repo-checkouts-obsidian-web-mcp.md",
        "type: canonical-state\n"
        "component_id: repo-checkouts-obsidian-web-mcp\n"
        "state: active\n"
        "content_updated: '2026-08-09'\n"
        "verified_at: '2026-08-18'\n"
        "source: 'stale claim: dev checkout is on main'\n",
    )
    checkers = {
        "repo-checkouts-obsidian-web-mcp": lambda record: check_repo_checkouts_obsidian_web_mcp(
            record, expected=drifted_expected
        )
    }
    findings = check_live_state_alignment(records_dir, checkers=checkers)
    assert any(f["issue"] == "branch_mismatch" for f in findings)


def test_check_live_state_alignment_no_record_produces_no_finding(tmp_path):
    """A registered component_id with no record on disk at all is the
    referential check's job (referenced_but_missing), not this one's."""
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    findings = check_live_state_alignment(records_dir)
    assert findings == []


def test_check_live_state_alignment_duplicate_authority_is_a_finding(tmp_path):
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    fm = (
        "type: canonical-state\n"
        "component_id: repo-checkouts-obsidian-web-mcp\n"
        "state: active\n"
        "content_updated: '2026-08-09'\n"
        "verified_at: '2026-08-20'\n"
        "source: 'test'\n"
    )
    _write_record(records_dir, "repo-checkouts-obsidian-web-mcp.md", fm)
    _write_record(records_dir, "repo-checkouts-obsidian-web-mcp--dup.md", fm)
    findings = check_live_state_alignment(records_dir)
    assert any(f["issue"] == "duplicate_authority" for f in findings)


def test_checker_exception_is_fail_closed(tmp_path):
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    _write_record(
        records_dir,
        "repo-checkouts-obsidian-web-mcp.md",
        "type: canonical-state\n"
        "component_id: repo-checkouts-obsidian-web-mcp\n"
        "state: active\n"
        "content_updated: '2026-08-09'\n"
        "verified_at: '2026-08-20'\n"
        "source: 'test'\n",
    )

    def _boom(record):
        raise RuntimeError("simulated inspection crash")

    findings = check_live_state_alignment(
        records_dir, checkers={"repo-checkouts-obsidian-web-mcp": _boom}
    )
    assert len(findings) == 1
    assert findings[0]["issue"] == "checker_error"
    assert "simulated inspection crash" in findings[0]["detail"]
