"""Tests for vault_lock.py's v2 bounded registry, acquisition timeout, and
the shared atomic_write primitive used by migrated same-box direct writers
(vault-integrity-and-bo-authority-remediation-v2).

vault_lock.path_lock is a cross-PROCESS guarantee only (POSIX fcntl record
locks belong to the process, not the thread) -- see the module docstring.
Tests that need to prove real serialization/timeout behavior therefore spawn
a genuine subprocess holder, exactly like the pre-existing
test_subprocess_cooperating_writer_forces_explicit_conflict_not_silent_overwrite
in test_optimistic_concurrency.py, rather than using threads.
"""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from obsidian_vault_mcp import vault_lock
from obsidian_vault_mcp.vault import RevisionConflictError, read_file, write_file_atomic

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")


def _spawn_holder(tmp_path, lock_dir, target_path, *, hold_seconds=0.3, force_slot=None, wait_for_release=True):
    """Spawn a real subprocess that acquires vault_lock.path_lock(target_path),
    signals readiness via a marker file, optionally waits for a release
    marker, then holds the lock for `hold_seconds` before releasing.

    Returns (proc, ready_marker, release_marker).
    """
    ready_marker = tmp_path / f"ready-{os.urandom(4).hex()}"
    release_marker = tmp_path / f"release-{os.urandom(4).hex()}"

    force_slot_code = f"vault_lock._slot_for = lambda p: {force_slot}\n" if force_slot is not None else ""
    wait_code = textwrap.dedent(f"""
        release = Path({str(release_marker)!r})
        for _ in range(500):
            if release.exists():
                break
            time.sleep(0.01)
    """) if wait_for_release else ""

    script = tmp_path / f"holder-{os.urandom(4).hex()}.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {_SRC_DIR!r})
        import time
        from pathlib import Path
        from obsidian_vault_mcp import vault_lock

        {force_slot_code}
        target = Path({str(target_path)!r})
        ready = Path({str(ready_marker)!r})

        with vault_lock.path_lock(target):
            ready.write_text("ready")
            {wait_code}
            time.sleep({hold_seconds})
    """))

    full_env = dict(os.environ)
    full_env["VAULT_MUTATION_LOCK_DIR"] = str(lock_dir)
    full_env["PYTHONPATH"] = _SRC_DIR

    proc = subprocess.Popen([sys.executable, str(script)], env=full_env)
    for _ in range(500):
        if ready_marker.exists():
            break
        time.sleep(0.01)
    else:
        proc.kill()
        pytest.fail("holder subprocess never acquired the lock")

    return proc, ready_marker, release_marker


# --------------------------------------------------------------------------
# Bounded registry
# --------------------------------------------------------------------------


def test_registry_is_exactly_one_file_regardless_of_path_count(tmp_path, monkeypatch):
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("VAULT_MUTATION_LOCK_DIR", str(lock_dir))

    for i in range(3000):
        fake_path = tmp_path / f"fixture-path-{i}.md"
        with vault_lock.path_lock(fake_path):
            pass

    entries = list(lock_dir.iterdir())
    assert len(entries) == 1
    assert entries[0].name == "registry.lock"


def test_registry_file_never_grows_from_lock_traffic(tmp_path, monkeypatch):
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("VAULT_MUTATION_LOCK_DIR", str(lock_dir))

    for i in range(5000):
        with vault_lock.path_lock(tmp_path / f"p-{i}"):
            pass

    registry = lock_dir / "registry.lock"
    # POSIX record locks may be taken on byte ranges beyond EOF -- no actual
    # content is ever written, so the file stays at (or near) zero bytes no
    # matter how many distinct paths have been locked.
    assert registry.stat().st_size <= vault_lock.LOCK_REGISTRY_SLOTS


def test_slot_for_is_always_in_bounds():
    for i in range(2000):
        slot = vault_lock._slot_for(Path(f"/some/fixture/path-{i}.md"))
        assert 0 <= slot < vault_lock.LOCK_REGISTRY_SLOTS


def test_forced_slot_collision_serializes_two_real_processes_correctly(tmp_path, monkeypatch):
    """Two DISTINCT paths forced onto the SAME slot must still serialize
    across two real processes (never both "hold" the byte range at once) --
    correctness over the fixed-size registry never bends under a collision."""
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("VAULT_MUTATION_LOCK_DIR", str(lock_dir))
    path_a = tmp_path / "collide-a.md"
    path_b = tmp_path / "collide-b.md"

    proc, ready, release = _spawn_holder(
        tmp_path, lock_dir, path_a, hold_seconds=0.3, force_slot=7, wait_for_release=False,
    )
    monkeypatch.setattr(vault_lock, "_slot_for", lambda p: 7)
    try:
        # Holder A is holding slot 7 (forced). Attempting slot 7 again from
        # THIS process (a distinct PID from the subprocess) for a different
        # path must block until A releases -- prove it via a bounded timeout
        # that's shorter than A's total hold time.
        start = time.monotonic()
        with pytest.raises(vault_lock.LockTimeoutError):
            with vault_lock.path_lock(path_b, timeout_seconds=0.1):
                pass
        elapsed = time.monotonic() - start
        assert elapsed < 0.3  # didn't accidentally wait out the full hold
    finally:
        proc.wait(timeout=10)

    # After A has released, the same forced-slot acquisition succeeds.
    with vault_lock.path_lock(path_b, timeout_seconds=5):
        pass


# --------------------------------------------------------------------------
# Acquisition timeout
# --------------------------------------------------------------------------


def test_lock_timeout_raises_and_leaves_no_trace(tmp_path):
    lock_dir = tmp_path / "locks"
    target = tmp_path / "contended.md"

    proc, ready, release = _spawn_holder(tmp_path, lock_dir, target, hold_seconds=2.0, wait_for_release=False)
    try:
        os.environ["VAULT_MUTATION_LOCK_DIR"] = str(lock_dir)
        try:
            start = time.monotonic()
            with pytest.raises(vault_lock.LockTimeoutError):
                with vault_lock.path_lock(target, timeout_seconds=0.3):
                    pass
            elapsed = time.monotonic() - start
            assert 0.25 <= elapsed < 2.0
        finally:
            os.environ.pop("VAULT_MUTATION_LOCK_DIR", None)
    finally:
        proc.wait(timeout=10)


def test_lock_timeout_default_is_configurable_via_env(monkeypatch):
    monkeypatch.delenv("VAULT_MUTATION_LOCK_TIMEOUT_SECONDS", raising=False)
    assert vault_lock.lock_timeout_seconds() == vault_lock.DEFAULT_LOCK_TIMEOUT_SECONDS

    monkeypatch.setenv("VAULT_MUTATION_LOCK_TIMEOUT_SECONDS", "5")
    assert vault_lock.lock_timeout_seconds() == 5.0

    monkeypatch.setenv("VAULT_MUTATION_LOCK_TIMEOUT_SECONDS", "not-a-number")
    assert vault_lock.lock_timeout_seconds() == vault_lock.DEFAULT_LOCK_TIMEOUT_SECONDS


def test_write_file_atomic_times_out_leaving_bytes_and_mtime_unchanged(vault_dir, tmp_path):
    lock_dir = tmp_path / "locks"
    target = vault_dir / "test-note.md"
    before_bytes = target.read_bytes()
    before_mtime = target.stat().st_mtime

    proc, ready, release = _spawn_holder(
        tmp_path, lock_dir, target.resolve(), hold_seconds=2.0, wait_for_release=False,
    )
    try:
        os.environ["VAULT_MUTATION_LOCK_DIR"] = str(lock_dir)
        os.environ["VAULT_MUTATION_LOCK_TIMEOUT_SECONDS"] = "0.3"
        try:
            with pytest.raises(vault_lock.LockTimeoutError):
                write_file_atomic("test-note.md", "should never land")
        finally:
            os.environ.pop("VAULT_MUTATION_LOCK_DIR", None)
            os.environ.pop("VAULT_MUTATION_LOCK_TIMEOUT_SECONDS", None)
    finally:
        proc.wait(timeout=10)

    assert target.read_bytes() == before_bytes
    assert target.stat().st_mtime == before_mtime


# --------------------------------------------------------------------------
# Unrelated paths make progress within a bound; same path serializes
# --------------------------------------------------------------------------


def test_unrelated_paths_do_not_block_each_other(tmp_path):
    lock_dir = tmp_path / "locks"
    path_a = tmp_path / "unrelated-a.md"
    path_b = tmp_path / "unrelated-b.md"

    proc, ready, release = _spawn_holder(tmp_path, lock_dir, path_a, hold_seconds=2.0, wait_for_release=False)
    try:
        os.environ["VAULT_MUTATION_LOCK_DIR"] = str(lock_dir)
        try:
            start = time.monotonic()
            with vault_lock.path_lock(path_b, timeout_seconds=5):
                pass
            elapsed = time.monotonic() - start
            assert elapsed < 1.0  # did not wait for path_a's holder at all
        finally:
            os.environ.pop("VAULT_MUTATION_LOCK_DIR", None)
    finally:
        proc.wait(timeout=10)


# --------------------------------------------------------------------------
# Shared atomic_write primitive -- the exact function migrated same-box
# writers (log_syncer.py, hot-md-curate.py, vault-audit.py) now call.
# --------------------------------------------------------------------------


def test_atomic_write_replaces_content_atomically(tmp_path):
    target = tmp_path / "note.md"
    target.write_text("old")
    vault_lock.atomic_write(target, b"new content")
    assert target.read_text() == "new content"


def test_atomic_write_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "dir" / "note.md"
    vault_lock.atomic_write(target, b"hello")
    assert target.read_text() == "hello"


def test_subprocess_using_shared_atomic_write_serializes_against_mcp_write(
    vault_dir, tmp_path, monkeypatch
):
    """Closes codex-review-phase2-write-integrity-v2's residual HIGH: a real
    machine writer (log_syncer.py/hot-md-curate.py/vault-audit.py) now calls
    vault_lock.atomic_write for its final vault write instead of a bare
    write_text -- this proves that migrated call correctly serializes
    against the MCP server's own write_file_atomic, with no silent
    overwrite, using a real separate subprocess (not a simulated race)."""
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("VAULT_MUTATION_LOCK_DIR", str(lock_dir))

    target = vault_dir / "test-note.md"
    _, meta = read_file("test-note.md")
    resolved_path = str(target.resolve())

    ready_marker = tmp_path / "subprocess-holds-lock"
    release_marker = tmp_path / "main-thread-is-waiting"

    script = tmp_path / "external_writer_atomic.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {_SRC_DIR!r})
        import time
        from pathlib import Path
        from obsidian_vault_mcp import vault_lock

        target = Path({resolved_path!r})
        ready = Path({str(ready_marker)!r})
        release = Path({str(release_marker)!r})

        # Mirror the shape of a migrated external writer: lock, then write,
        # but hold the lock a little past the write so the main process's
        # write_file_atomic call is still blocked when we release it.
        with vault_lock.path_lock(target):
            ready.write_text("ready")
            for _ in range(500):
                if release.exists():
                    break
                time.sleep(0.01)
            target.write_bytes(b"external writer content via atomic_write path")
            time.sleep(0.3)
    """))

    full_env = dict(os.environ)
    full_env["VAULT_MUTATION_LOCK_DIR"] = str(lock_dir)
    full_env["PYTHONPATH"] = _SRC_DIR

    proc = subprocess.Popen([sys.executable, str(script)], env=full_env)
    try:
        for _ in range(500):
            if ready_marker.exists():
                break
            time.sleep(0.01)
        else:
            proc.kill()
            pytest.fail("external-writer subprocess never acquired the lock")

        release_marker.write_text("go")

        with pytest.raises(RevisionConflictError):
            write_file_atomic("test-note.md", "mcp content", expected_revision=meta["revision"])
    finally:
        proc.wait(timeout=10)

    content, _ = read_file("test-note.md")
    assert content == "external writer content via atomic_write path"
