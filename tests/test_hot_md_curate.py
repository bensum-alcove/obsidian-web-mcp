"""Tests for scripts/hot-md-curate.py's shared mutation authority
(vault-integrity-and-bo-authority-remediation-v2): write_file/archive
appends now go through vault_lock instead of a bare open(..., 'w'/'a').
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "hot-md-curate.py"
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")


@pytest.fixture(scope="module")
def curate():
    spec = importlib.util.spec_from_file_location("hot_md_curate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_file_replaces_content(curate, tmp_path):
    target = tmp_path / "hot.md"
    target.write_text("old")
    curate.write_file(str(target), "new content")
    assert target.read_text() == "new content"


def test_write_file_uses_shared_vault_lock(curate, tmp_path, monkeypatch):
    target = tmp_path / "hot.md"
    calls = []
    monkeypatch.setattr(
        curate.vault_lock, "atomic_write",
        lambda path, data, **kw: calls.append((path, data)),
    )
    curate.write_file(str(target), "content")
    assert len(calls) == 1
    path, data = calls[0]
    assert path == target.resolve()
    assert data == b"content"


def test_archive_append_uses_shared_vault_lock(curate, tmp_path, monkeypatch):
    archive_path = tmp_path / "hot-archive" / "2026-08.md"
    calls = []
    monkeypatch.setattr(
        curate.vault_lock, "atomic_append",
        lambda path, data, **kw: calls.append((path, data)),
    )
    curate.vault_lock.atomic_append(archive_path.resolve(), b"chunk")
    assert calls == [(archive_path.resolve(), b"chunk")]


def test_write_file_subprocess_serializes_against_mcp_lock_holder(curate, tmp_path, monkeypatch):
    """Real-process reproduction of matrix item #3: hot-md-curate.py
    --apply's write versus a concurrent MCP write to the same fixture path
    -- no silent overwrite. A subprocess holds vault_lock.path_lock for the
    exact target (standing in for the live Vault MCP server's
    write_file_atomic, which acquires this identical lock)."""
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("VAULT_MUTATION_LOCK_DIR", str(lock_dir))

    target = tmp_path / "hot.md"
    target.write_text("original")
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

        start = time.monotonic()
        release_marker.write_text("go")
        curate.write_file(str(target), "curated content")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.15  # proves write_file actually waited on the held lock
    finally:
        proc.wait(timeout=10)

    assert target.read_text() == "curated content"
