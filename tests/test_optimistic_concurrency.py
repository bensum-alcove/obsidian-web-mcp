"""Race and contract tests for optimistic-concurrency (revision-guarded) writes."""

import json
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from obsidian_vault_mcp import vault as v
from obsidian_vault_mcp import vault_lock
from obsidian_vault_mcp.tools.manage import vault_delete, vault_move
from obsidian_vault_mcp.tools.read import vault_read
from obsidian_vault_mcp.tools.write import (
    vault_append,
    vault_batch_frontmatter_update,
    vault_batch_str_replace,
    vault_batch_write,
    vault_patch_section,
    vault_str_replace,
    vault_write,
)
from obsidian_vault_mcp.vault import (
    RevisionConflictError,
    compute_revision,
    delete_path,
    move_path,
    read_file,
    write_file_atomic,
)


# --------------------------------------------------------------------------
# compute_revision / read_file metadata
# --------------------------------------------------------------------------


def test_compute_revision_is_pure_function_of_bytes():
    a = compute_revision(b"hello")
    b = compute_revision(b"hello")
    c = compute_revision(b"world")
    assert a == b
    assert a != c


def test_compute_revision_absent_sentinel_for_missing_file():
    assert compute_revision(None) == "absent"


def test_read_file_exposes_revision_matching_disk_bytes(vault_dir):
    content, metadata = read_file("test-note.md")
    assert metadata["revision"] == compute_revision((vault_dir / "test-note.md").read_bytes())


def test_two_reads_of_unchanged_file_get_same_revision(vault_dir):
    _, m1 = read_file("test-note.md")
    _, m2 = read_file("test-note.md")
    assert m1["revision"] == m2["revision"]


def test_revision_changes_after_a_write(vault_dir):
    _, before = read_file("test-note.md")
    write_file_atomic("test-note.md", "new body")
    _, after = read_file("test-note.md")
    assert before["revision"] != after["revision"]


def test_read_file_opens_the_target_exactly_once(vault_dir, monkeypatch):
    """codex-review-phase2-write-integrity HIGH #1: content and revision used
    to come from separate stat()/read_text()/read_bytes() calls, leaving a
    seam for a writer to land in between and produce an incoherent pair.
    read_file() must now open the file exactly once and derive both values
    from that single read -- there should be no second syscall left for a
    writer to land between.
    """
    import os as os_module

    real_open = os_module.open
    open_calls = []

    def counting_open(path, *a, **k):
        open_calls.append(path)
        return real_open(path, *a, **k)

    monkeypatch.setattr(os_module, "open", counting_open)
    read_file("test-note.md")
    target_opens = [c for c in open_calls if str(c).endswith("test-note.md")]
    assert len(target_opens) == 1


def test_read_file_content_and_revision_always_agree_under_concurrent_writes(vault_dir):
    """Stress-test the coherent-snapshot guarantee: a background thread keeps
    replacing the file's content in a tight loop while the foreground thread
    repeatedly calls read_file(). For every single read, the returned
    content's hash must equal the returned revision -- by construction, not
    by luck -- since both now come from one read of one file descriptor.
    """
    stop = threading.Event()
    counter = {"n": 0}

    def writer():
        while not stop.is_set():
            counter["n"] += 1
            write_file_atomic("test-note.md", f"version {counter['n']}")

    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(200):
            content, metadata = read_file("test-note.md")
            assert metadata["revision"] == compute_revision(content.encode("utf-8")), (
                "content and revision must always come from the same on-disk snapshot"
            )
    finally:
        stop.set()
        t.join()


# --------------------------------------------------------------------------
# Core CAS semantics at the write_file_atomic choke point
# --------------------------------------------------------------------------


def test_matching_expected_revision_succeeds(vault_dir):
    _, meta = read_file("test-note.md")
    write_file_atomic("test-note.md", "updated body", expected_revision=meta["revision"])
    content, _ = read_file("test-note.md")
    assert content == "updated body"


def test_stale_expected_revision_raises_conflict_and_leaves_file_untouched(vault_dir):
    path = vault_dir / "test-note.md"
    _, meta = read_file("test-note.md")
    # Someone else writes first, advancing the revision.
    write_file_atomic("test-note.md", "written by someone else")

    original_bytes = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns

    with pytest.raises(RevisionConflictError) as excinfo:
        write_file_atomic("test-note.md", "my stale write", expected_revision=meta["revision"])

    assert excinfo.value.expected_revision == meta["revision"]
    assert excinfo.value.current_revision == compute_revision(original_bytes)
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime


def test_no_expected_revision_is_unprotected_legacy_behavior(vault_dir):
    """Callers that never migrate keep writing straight through, no matter who else wrote."""
    write_file_atomic("test-note.md", "first")
    write_file_atomic("test-note.md", "second")  # no expected_revision at all
    content, _ = read_file("test-note.md")
    assert content == "second"


def test_expected_revision_absent_allows_creating_new_file(vault_dir):
    is_new, _ = write_file_atomic("brand-new.md", "hello", expected_revision="absent")
    assert is_new is True


def test_expected_revision_absent_conflicts_if_file_now_exists(vault_dir):
    write_file_atomic("now-exists.md", "surprise")
    with pytest.raises(RevisionConflictError):
        write_file_atomic("now-exists.md", "clobber", expected_revision="absent")


def test_same_content_no_op_succeeds_despite_stale_revision(vault_dir):
    """Re-committing exactly the bytes already on disk carries no data-loss risk."""
    _, meta = read_file("test-note.md")
    write_file_atomic("test-note.md", "content that will stick")
    # meta.revision is now stale, but writing the SAME content that's already there
    # (what write_file_atomic will see as current) must not conflict.
    current_content, current_meta = read_file("test-note.md")
    write_file_atomic("test-note.md", current_content, expected_revision=meta["revision"])
    content, _ = read_file("test-note.md")
    assert content == current_content


# --------------------------------------------------------------------------
# "Two readers, same revision" race: A succeeds, B conflicts
# --------------------------------------------------------------------------


def test_two_readers_same_revision_first_writer_wins(vault_dir):
    _, reader_a = read_file("test-note.md")
    _, reader_b = read_file("test-note.md")
    assert reader_a["revision"] == reader_b["revision"]

    write_file_atomic("test-note.md", "A's change", expected_revision=reader_a["revision"])

    with pytest.raises(RevisionConflictError):
        write_file_atomic("test-note.md", "B's conflicting change", expected_revision=reader_b["revision"])

    content, _ = read_file("test-note.md")
    assert content == "A's change"


def test_concurrent_threads_same_path_exactly_one_wins(vault_dir):
    """Real threads racing on the same path with the same expected_revision:
    the per-path lock must serialize them so exactly one succeeds."""
    _, meta = read_file("test-note.md")
    barrier = threading.Barrier(2)
    outcomes = []
    outcomes_lock = threading.Lock()

    def attempt(label):
        barrier.wait()
        try:
            write_file_atomic("test-note.md", f"written by {label}", expected_revision=meta["revision"])
            with outcomes_lock:
                outcomes.append(("ok", label))
        except RevisionConflictError:
            with outcomes_lock:
                outcomes.append(("conflict", label))

    threads = [threading.Thread(target=attempt, args=(f"thread-{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = [o for o in outcomes if o[0] == "ok"]
    conflicts = [o for o in outcomes if o[0] == "conflict"]
    assert len(successes) == 1
    assert len(conflicts) == 1


def test_subprocess_cooperating_writer_forces_explicit_conflict_not_silent_overwrite(vault_dir, tmp_path, monkeypatch):
    """codex-review-phase2-write-integrity HIGH #2 reproduction target:
    "paused immediately before os.replace, wrote external directly to the
    target after the expected revision had passed, then resumed. The guarded
    MCP write returned success and the final content was mcp; the
    intervening external version was silently lost." That probe used a raw
    filesystem write with zero coordination -- vault_lock.py's own docstring
    is explicit that its process-shared lock only protects a writer that
    ALSO acquires it (a "cooperating" writer, the framing this build's spec
    uses), not an arbitrary uncoordinated one.

    This test proves the mechanism holds for exactly that class: a REAL
    separate process that cooperates via vault_lock.path_lock, holds the
    lock, mutates the target directly, and only then releases it. The
    guarded write in THIS process must block until the lock is released and
    then raise RevisionConflictError (never silently overwrite) -- and the
    external writer's bytes must survive on disk.
    """
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("VAULT_MUTATION_LOCK_DIR", str(lock_dir))

    target = vault_dir / "test-note.md"
    _, meta = read_file("test-note.md")
    resolved_path = str(target.resolve())

    ready_marker = tmp_path / "subprocess-holds-lock"
    release_marker = tmp_path / "main-thread-is-waiting"

    script = tmp_path / "external_writer.py"
    script.write_text(textwrap.dedent(f"""
        import time
        from pathlib import Path
        from obsidian_vault_mcp import vault_lock

        target = Path({resolved_path!r})
        ready = Path({str(ready_marker)!r})
        release = Path({str(release_marker)!r})

        with vault_lock.path_lock(target):
            ready.write_text("ready")
            for _ in range(500):
                if release.exists():
                    break
                time.sleep(0.01)
            target.write_bytes(b"external writer content")
            time.sleep(0.3)  # keep holding the lock so the main write is still blocked on it
    """))

    env = {"VAULT_MUTATION_LOCK_DIR": str(lock_dir), "PYTHONPATH": str(
        __import__("pathlib").Path(__file__).resolve().parent.parent / "src"
    )}
    import os as os_module
    full_env = dict(os_module.environ)
    full_env.update(env)

    proc = subprocess.Popen([sys.executable, str(script)], env=full_env)
    try:
        for _ in range(500):
            if ready_marker.exists():
                break
            time.sleep(0.01)
        else:
            proc.kill()
            pytest.fail("external-writer subprocess never acquired the lock")

        # Give the main thread's write_file_atomic call a moment to actually
        # start blocking on the same lock before telling the subprocess it
        # may proceed with its write.
        release_marker.write_text("go")

        with pytest.raises(RevisionConflictError):
            write_file_atomic("test-note.md", "mcp content", expected_revision=meta["revision"])
    finally:
        proc.wait(timeout=10)

    content, _ = read_file("test-note.md")
    assert content == "external writer content"  # the cooperating external writer's bytes survive intact


# --------------------------------------------------------------------------
# Delete-recreate race
# --------------------------------------------------------------------------


def test_delete_recreate_race_is_a_conflict_not_silent_overwrite(vault_dir):
    _, reader = read_file("test-note.md")

    # File is deleted and a different file recreated at the same path in between.
    delete_path("test-note.md")
    write_file_atomic("test-note.md", "recreated with different content")

    with pytest.raises(RevisionConflictError):
        write_file_atomic("test-note.md", "stale writer's content", expected_revision=reader["revision"])

    content, _ = read_file("test-note.md")
    assert content == "recreated with different content"


def test_delete_with_stale_revision_conflicts(vault_dir):
    path = vault_dir / "test-note.md"
    _, reader = read_file("test-note.md")
    write_file_atomic("test-note.md", "changed after read")

    with pytest.raises(RevisionConflictError):
        delete_path("test-note.md", expected_revision=reader["revision"])

    assert path.exists()


def test_delete_with_matching_revision_succeeds(vault_dir):
    path = vault_dir / "test-note.md"
    _, reader = read_file("test-note.md")
    delete_path("test-note.md", expected_revision=reader["revision"])
    assert not path.exists()


def test_move_with_stale_revision_conflicts(vault_dir):
    src = vault_dir / "test-note.md"
    _, reader = read_file("test-note.md")
    write_file_atomic("test-note.md", "changed after read")

    with pytest.raises(RevisionConflictError):
        move_path("test-note.md", "moved-note.md", expected_revision=reader["revision"])

    assert src.exists()
    assert not (vault_dir / "moved-note.md").exists()


def test_move_with_matching_revision_succeeds(vault_dir):
    _, reader = read_file("test-note.md")
    move_path("test-note.md", "moved-note.md", expected_revision=reader["revision"])
    assert (vault_dir / "moved-note.md").exists()
    assert not (vault_dir / "test-note.md").exists()


def test_concurrent_moves_to_same_destination_exactly_one_wins_loser_source_intact(vault_dir):
    """codex-review-phase2-write-integrity HIGH #3 reproduction target:
    "Two moves from different sources therefore hold different locks, both
    observe an absent destination, and race at rename/replace ... Both calls
    returned True; both sources disappeared; only one source's bytes
    remained." Source and destination are now locked together (in
    deterministic order) for the whole operation, so exactly one of two
    concurrent movers to the same destination must succeed and the other
    must fail explicitly with its own source left untouched.
    """
    (vault_dir / "source-a.md").write_text("content from a")
    (vault_dir / "source-b.md").write_text("content from b")

    barrier = threading.Barrier(2)
    outcomes = []
    outcomes_lock = threading.Lock()

    def attempt(source, label):
        barrier.wait()
        try:
            move_path(source, "dest.md")
            with outcomes_lock:
                outcomes.append(("ok", label))
        except FileExistsError:
            with outcomes_lock:
                outcomes.append(("exists", label))

    threads = [
        threading.Thread(target=attempt, args=("source-a.md", "a")),
        threading.Thread(target=attempt, args=("source-b.md", "b")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = [o for o in outcomes if o[0] == "ok"]
    losers = [o for o in outcomes if o[0] == "exists"]
    assert len(successes) == 1
    assert len(losers) == 1

    # Exactly one source disappeared (the winner's); the loser's source must
    # still be present and untouched -- not silently lost.
    loser_label = losers[0][1]
    loser_source = vault_dir / f"source-{loser_label}.md"
    assert loser_source.exists()
    assert loser_source.read_text() == f"content from {loser_label}"

    winner_label = successes[0][1]
    winner_source = vault_dir / f"source-{winner_label}.md"
    assert not winner_source.exists()

    assert (vault_dir / "dest.md").read_text() == f"content from {winner_label}"


def test_expected_revision_rejected_for_directory_move(vault_dir):
    with pytest.raises(ValueError, match="directories"):
        move_path("subfolder", "subfolder-moved", expected_revision="absent")


def test_expected_revision_rejected_for_directory_delete(vault_dir):
    (vault_dir / "empty-dir").mkdir()
    with pytest.raises(ValueError, match="directories"):
        delete_path("empty-dir", expected_revision="absent")


# --------------------------------------------------------------------------
# Frontmatter update race (vault_batch_frontmatter_update)
# --------------------------------------------------------------------------


def test_batch_frontmatter_update_race(vault_dir):
    _, reader_a = read_file("test-note.md")
    _, reader_b = read_file("test-note.md")

    result_a = json.loads(vault_batch_frontmatter_update(
        [{"path": "test-note.md", "fields": {"status": "done"}, "expected_revision": reader_a["revision"]}]
    ))
    assert result_a["results"][0]["updated"] is True

    result_b = json.loads(vault_batch_frontmatter_update(
        [{"path": "test-note.md", "fields": {"status": "archived"}, "expected_revision": reader_b["revision"]}]
    ))
    assert result_b["results"][0]["updated"] is False
    assert result_b["results"][0]["conflict"] is True
    assert "expected_revision" in result_b["results"][0]
    assert "current_revision" in result_b["results"][0]


def test_batch_frontmatter_update_without_revision_is_unprotected(vault_dir):
    result = json.loads(vault_batch_frontmatter_update(
        [{"path": "test-note.md", "fields": {"status": "done"}}]
    ))
    assert result["results"][0]["updated"] is True


# --------------------------------------------------------------------------
# patch/str-replace race
# --------------------------------------------------------------------------


def test_patch_section_race(vault_dir):
    (vault_dir / "sections.md").write_text("# Target\n\noriginal\n")
    _, reader_a = read_file("sections.md")
    _, reader_b = read_file("sections.md")

    result_a = json.loads(vault_patch_section("sections.md", "# Target", "A's edit", reader_a["revision"]))
    assert "error" not in result_a

    result_b = json.loads(vault_patch_section("sections.md", "# Target", "B's edit", reader_b["revision"]))
    assert result_b.get("conflict") is True
    assert "A's edit" in (vault_dir / "sections.md").read_text()


def test_str_replace_race(vault_dir):
    (vault_dir / "replace.md").write_text("the quick brown fox\n")
    _, reader_a = read_file("replace.md")
    _, reader_b = read_file("replace.md")

    result_a = json.loads(vault_str_replace("replace.md", "quick", "slow", False, reader_a["revision"]))
    assert "error" not in result_a

    result_b = json.loads(vault_str_replace("replace.md", "brown", "red", False, reader_b["revision"]))
    assert result_b.get("conflict") is True
    assert "the slow brown fox" in (vault_dir / "replace.md").read_text()


def test_batch_str_replace_race(vault_dir):
    (vault_dir / "batch-replace.md").write_text("alpha beta\n")
    _, reader_a = read_file("batch-replace.md")
    _, reader_b = read_file("batch-replace.md")

    json.loads(vault_batch_str_replace(
        [{"path": "batch-replace.md", "old_str": "alpha", "new_str": "ALPHA", "expected_revision": reader_a["revision"]}]
    ))
    result_b = json.loads(vault_batch_str_replace(
        [{"path": "batch-replace.md", "old_str": "beta", "new_str": "BETA", "expected_revision": reader_b["revision"]}]
    ))
    assert result_b["results"][0].get("conflict") is True
    assert "ALPHA beta" in (vault_dir / "batch-replace.md").read_text()


def test_batch_write_race(vault_dir):
    (vault_dir / "batch-write.md").write_text("v0\n")
    _, reader_a = read_file("batch-write.md")
    _, reader_b = read_file("batch-write.md")

    json.loads(vault_batch_write([{"path": "batch-write.md", "content": "v1", "expected_revision": reader_a["revision"]}]))
    result_b = json.loads(vault_batch_write(
        [{"path": "batch-write.md", "content": "v2-stale", "expected_revision": reader_b["revision"]}]
    ))
    assert result_b["results"][0]["written"] is False
    assert result_b["results"][0]["conflict"] is True


def test_append_race(vault_dir):
    (vault_dir / "append.md").write_text("base\n")
    _, reader_a = read_file("append.md")
    _, reader_b = read_file("append.md")

    result_a = json.loads(vault_append("append.md", "A\n", True, reader_a["revision"]))
    assert "error" not in result_a

    result_b = json.loads(vault_append("append.md", "B\n", True, reader_b["revision"]))
    assert result_b.get("conflict") is True


def test_vault_write_tool_surfaces_conflict_fields(vault_dir):
    _, reader = read_file("test-note.md")
    write_file_atomic("test-note.md", "advanced by someone else")

    result = json.loads(vault_write("test-note.md", "stale content", True, False, reader["revision"]))
    assert result["conflict"] is True
    assert result["expected_revision"] == reader["revision"]
    assert "current_revision" in result
    assert "error" in result


def test_vault_move_tool_surfaces_conflict_fields(vault_dir):
    _, reader = read_file("test-note.md")
    write_file_atomic("test-note.md", "advanced")
    result = json.loads(vault_move("test-note.md", "elsewhere.md", True, reader["revision"]))
    assert result["conflict"] is True


def test_vault_delete_tool_surfaces_conflict_fields(vault_dir):
    _, reader = read_file("test-note.md")
    write_file_atomic("test-note.md", "advanced")
    result = json.loads(vault_delete("test-note.md", True, reader["revision"]))
    assert result["conflict"] is True


def test_batch_delete_expected_revisions_map(vault_dir):
    _, reader = read_file("test-note.md")
    write_file_atomic("test-note.md", "advanced")

    from obsidian_vault_mcp.tools.manage import vault_batch_delete

    result = json.loads(vault_batch_delete(
        ["test-note.md", "no-frontmatter.md"], True, {"test-note.md": reader["revision"]}
    ))
    by_path = {r["path"]: r for r in result["results"]}
    assert by_path["test-note.md"]["deleted"] is False
    assert by_path["test-note.md"]["conflict"] is True
    # no-frontmatter.md wasn't in the expected_revisions map -- unprotected, deletes fine.
    assert by_path["no-frontmatter.md"]["deleted"] is True


def test_end_to_end_via_vault_read_tool_returns_usable_revision(vault_dir):
    read_result = json.loads(vault_read("test-note.md"))
    revision = read_result["metadata"]["revision"]

    write_file_atomic("test-note.md", "concurrent change")

    write_result = json.loads(vault_write("test-note.md", "stale", True, False, revision))
    assert write_result["conflict"] is True


# --------------------------------------------------------------------------
# Mode plumbing: off / shadow / enforce
# --------------------------------------------------------------------------


def test_mode_defaults_to_enforce(monkeypatch):
    monkeypatch.delenv("VAULT_OPTIMISTIC_CONCURRENCY_MODE", raising=False)
    assert v._concurrency_mode() == "enforce"


def test_unrecognised_mode_falls_back_to_enforce(monkeypatch):
    monkeypatch.setenv("VAULT_OPTIMISTIC_CONCURRENCY_MODE", "bogus")
    assert v._concurrency_mode() == "enforce"


def test_off_mode_ignores_stale_revision_entirely(vault_dir, monkeypatch):
    monkeypatch.setenv("VAULT_OPTIMISTIC_CONCURRENCY_MODE", "off")
    _, reader = read_file("test-note.md")
    write_file_atomic("test-note.md", "advanced")
    # Would conflict under enforce/shadow, but "off" skips the check entirely.
    write_file_atomic("test-note.md", "clobbered anyway", expected_revision=reader["revision"])
    content, _ = read_file("test-note.md")
    assert content == "clobbered anyway"


def test_shadow_mode_logs_but_never_blocks(vault_dir, monkeypatch, caplog):
    monkeypatch.setenv("VAULT_OPTIMISTIC_CONCURRENCY_MODE", "shadow")
    _, reader = read_file("test-note.md")
    write_file_atomic("test-note.md", "advanced")

    with caplog.at_level("WARNING"):
        write_file_atomic("test-note.md", "shadow write proceeds", expected_revision=reader["revision"])

    content, _ = read_file("test-note.md")
    assert content == "shadow write proceeds"
    assert any("vault-concurrency[shadow]" in r.message for r in caplog.records)


def test_enforce_mode_blocks(vault_dir, monkeypatch):
    monkeypatch.setenv("VAULT_OPTIMISTIC_CONCURRENCY_MODE", "enforce")
    _, reader = read_file("test-note.md")
    write_file_atomic("test-note.md", "advanced")
    with pytest.raises(RevisionConflictError):
        write_file_atomic("test-note.md", "blocked", expected_revision=reader["revision"])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_metrics_track_unprotected_protected_and_conflicted(vault_dir):
    before = v.concurrency_metrics()

    write_file_atomic("metrics-a.md", "no revision at all")  # unprotected

    _, meta = read_file("test-note.md")
    write_file_atomic("test-note.md", "protected write", expected_revision=meta["revision"])  # protected

    with pytest.raises(RevisionConflictError):
        write_file_atomic("test-note.md", "stale", expected_revision=meta["revision"])  # conflicted

    after = v.concurrency_metrics()
    assert after["unprotected"] > before["unprotected"]
    assert after["protected"] > before["protected"]
    assert after["conflicted"] > before["conflicted"]
