"""Tests for scripts/log-rollover.py -- scheduled Write Rule 21 log rollover (dry-run by default)."""

import hashlib
import importlib.util
from datetime import date
from pathlib import Path

import pytest

from obsidian_vault_mcp import bounded_files

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "log-rollover.py"
LOG = "BS 2nd Brain/Alcove/Infrastructure/infrastructure-changelog.md"
LOG2 = "BS 2nd Brain/_log.md"
TODAY = date(2026, 11, 1)


@pytest.fixture
def rollover():
    spec = importlib.util.spec_from_file_location("log_rollover", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log_text(pad: int = 0) -> str:
    return (
        "---\ntype: changelog\nread_policy: section-only\nupdated: \"2026-10-07\"\n---\n"
        "# Infrastructure Changelog\n\n"
        "## 2026-10-07 — Newest\nbody\n\n## 2026-09-01 — Middle\nbody\n\n## 2026-08-15 — Oldest\n"
        + ("x" * pad)
    )


def _write(vault_dir, rel, text):
    p = vault_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_default_is_dry_run_and_changes_nothing(rollover, vault_dir, capsys):
    p = _write(vault_dir, LOG, _log_text(bounded_files.LOG_ROLLOVER_BYTES))
    before = p.read_bytes()
    assert rollover.main([]) == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out and "would roll" in out
    assert p.read_bytes() == before
    assert [f.name for f in p.parent.iterdir()] == [p.name]


def test_apply_rolls_byte_for_byte_and_starts_new_volume(rollover, vault_dir):
    p = _write(vault_dir, LOG, _log_text(bounded_files.LOG_ROLLOVER_BYTES))
    original_sha = _sha(p)

    results = rollover.run(mode="if-due", apply=True, today=TODAY)

    rolled = vault_dir / "BS 2nd Brain/Alcove/Infrastructure/infrastructure-changelog-2026-08-15-to-2026-10-07.md"
    assert results[0]["action"] == "rolled"
    assert rolled.exists() and _sha(rolled) == original_sha
    new = p.read_text()
    assert 'continued_from: "BS 2nd Brain/Alcove/Infrastructure/infrastructure-changelog-2026-08-15-to-2026-10-07.md"' in new
    assert 'read_policy: "section-only"' in new
    assert 'type: "changelog"' in new
    assert "## 2026-11-01 — Log rolled over" in new
    assert "sha256:" in new
    assert p.stat().st_size < 2000


def test_small_current_log_is_not_due(rollover, vault_dir):
    p = _write(vault_dir, LOG, _log_text())
    before = p.read_bytes()
    results = rollover.run(mode="if-due", apply=True, today=TODAY)
    assert results[0]["action"] == "not-due"
    assert p.read_bytes() == before


def test_monthly_rolls_prior_month_volume_once(rollover, vault_dir):
    p = _write(vault_dir, LOG, _log_text())
    first = rollover.run(mode="monthly", apply=True, today=TODAY)
    assert first[0]["action"] == "rolled"
    # The fresh volume only holds current-month entries, so a rerun does nothing.
    again = rollover.run(mode="monthly", apply=True, today=TODAY)
    assert again[0]["action"] == "not-due"
    assert len(list(p.parent.glob("infrastructure-changelog-*.md"))) == 1


def test_revision_conflict_aborts_before_anything_moves(rollover, vault_dir):
    p = _write(vault_dir, LOG, _log_text(bounded_files.LOG_ROLLOVER_BYTES))
    plan = rollover.plan_roll(LOG, "if-due", TODAY)
    with p.open("a") as f:
        f.write("\n## 2026-10-08 — written after the plan was made\n")
    after_edit = p.read_bytes()
    with pytest.raises(Exception):
        rollover.apply_roll(plan, TODAY)
    assert p.read_bytes() == after_edit
    assert [f.name for f in p.parent.iterdir()] == [p.name]


def test_failed_new_volume_write_restores_original(rollover, vault_dir, monkeypatch):
    p = _write(vault_dir, LOG, _log_text(bounded_files.LOG_ROLLOVER_BYTES))
    before = p.read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(rollover.vault, "write_file_atomic", boom)
    results = rollover.run(mode="if-due", apply=True, today=TODAY)
    assert results[0]["action"] == "error"
    assert p.read_bytes() == before
    assert [f.name for f in p.parent.iterdir()] == [p.name]


def test_never_touches_hot_md_and_skips_missing_logs(rollover, vault_dir):
    hot = _write(vault_dir, "BS 2nd Brain/Alcove/Infrastructure/hot.md", "x" * 900_000)
    log2 = _write(vault_dir, LOG2, "# Log\n\n## 2026-10-01 — entry\n")
    hot_before, log2_before = hot.read_bytes(), log2.read_bytes()

    results = rollover.run(mode="if-due", apply=True, today=TODAY)

    assert {r["path"] for r in results} == {LOG2}
    assert hot.read_bytes() == hot_before
    assert log2.read_bytes() == log2_before
