"""Tests for /home/ben_sum/vault-audit.py's shared mutation authority
(vault-integrity-and-bo-authority-remediation-v2). This script lives outside
any git repository (a standalone fortnightly cron target) but already
imports obsidian_vault_mcp.frontmatter_safe from THIS checkout via sys.path
-- it now also imports obsidian_vault_mcp.vault_lock the same way, so its
canonical-Markdown writes (--autofix) and its append-only report writes go
through the same shared cross-process mutation authority as the live Vault
MCP server and the other migrated same-box writers.
"""

import datetime
import importlib.util
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

SCRIPT_PATH = Path("/home/ben_sum/vault-audit.py")
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")

pytestmark = pytest.mark.skipif(
    not SCRIPT_PATH.exists(), reason="vault-audit.py not present on this host"
)


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("vault_audit_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_autofix_uses_shared_vault_lock(audit, tmp_path, monkeypatch):
    target = tmp_path / "note.md"
    target.write_text("---\nupdated: 2026-01-01\n---\n\nbody")

    calls = []
    monkeypatch.setattr(
        audit.vault_lock, "atomic_write",
        lambda path, data, **kw: calls.append((path, data)),
    )
    stale_item = {
        "rp": "note.md", "fp": str(target),
        "upd_date": datetime.date(2026, 1, 1), "mtime_date": datetime.date(2026, 2, 1),
    }
    fixed, skipped, errored = audit.autofix_stale_frontmatter([stale_item])
    assert fixed and not errored
    assert len(calls) == 1
    path, data = calls[0]
    assert path == target.resolve()
    assert b"updated: '2026-02-01'" in data or b"updated: 2026-02-01" in data


def test_report_append_uses_shared_vault_lock(audit, tmp_path, monkeypatch):
    report_path = tmp_path / "vault-audit-log.md"
    monkeypatch.setattr(audit, "REPORT_FILE", str(report_path))
    monkeypatch.setattr(audit, "VAULT_ROOT", str(tmp_path / "empty-vault"))
    os.makedirs(audit.VAULT_ROOT, exist_ok=True)
    monkeypatch.setattr(audit, "TELEGRAM_TOKEN", "")

    calls = []
    monkeypatch.setattr(
        audit.vault_lock, "atomic_append",
        lambda path, data, **kw: calls.append((path, data)),
    )
    audit.run_audit(autofix=False)
    assert len(calls) == 1
    path, data = calls[0]
    assert path == report_path.resolve()
    assert b"Vault Health Audit" in data


def test_autofix_subprocess_serializes_against_mcp_lock_holder(audit, tmp_path, monkeypatch):
    """Real-process reproduction of matrix item #4: vault-audit.py
    --autofix versus a concurrent MCP write to the same fixture path -- no
    silent overwrite."""
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("VAULT_MUTATION_LOCK_DIR", str(lock_dir))

    target = tmp_path / "note.md"
    target.write_text("---\nupdated: 2026-01-01\n---\n\nbody")
    resolved = target.resolve()

    ready_marker = tmp_path / "holder-ready"
    release_marker = tmp_path / "holder-release"

    script = tmp_path / "mcp_holder.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {SRC_DIR!r})
        import time
        from pathlib import Path
        from obsidian_vault_mcp import vault_lock

        target = Path({str(resolved)!r})
        ready = Path({str(ready_marker)!r})
        release = Path({str(release_marker)!r})

        with vault_lock.path_lock(target):
            ready.write_text("ready")
            for _ in range(500):
                if release.exists():
                    break
                time.sleep(0.01)
            time.sleep(0.2)
    """))

    full_env = dict(os.environ)
    full_env["VAULT_MUTATION_LOCK_DIR"] = str(lock_dir)
    full_env["PYTHONPATH"] = SRC_DIR

    proc = subprocess.Popen([sys.executable, str(script)], env=full_env)
    try:
        for _ in range(500):
            if ready_marker.exists():
                break
            time.sleep(0.01)
        else:
            proc.kill()
            pytest.fail("MCP-holder subprocess never acquired the lock")

        stale_item = {
            "rp": "note.md", "fp": str(target),
            "upd_date": datetime.date(2026, 1, 1), "mtime_date": datetime.date(2026, 2, 1),
        }
        start = time.monotonic()
        release_marker.write_text("go")
        fixed, skipped, errored = audit.autofix_stale_frontmatter([stale_item])
        elapsed = time.monotonic() - start
        assert elapsed >= 0.15
        assert fixed and not errored
    finally:
        proc.wait(timeout=10)

    assert "updated: '2026-02-01'" in target.read_text() or "updated: 2026-02-01" in target.read_text()
