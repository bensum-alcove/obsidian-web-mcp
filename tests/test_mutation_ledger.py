"""Schema, rotation, and behavioural contract tests for the mutation ledger."""

import json

import pytest

from obsidian_vault_mcp import mutation_ledger as ml
from obsidian_vault_mcp.tools.manage import vault_delete, vault_list, vault_move
from obsidian_vault_mcp.vault import (
    RevisionConflictError,
    delete_path,
    move_path,
    read_file,
    write_file_atomic,
)


@pytest.fixture(autouse=True)
def _reset_ledger_handler():
    """Close and drop any handler left open by a previous test.

    Each test that exercises the ledger uses its own tmp_path-derived ledger
    dir, so this is about promptly releasing file handles, not test
    isolation of content (query_events always reads from the path a test
    passes explicitly).
    """
    yield
    for handler in list(ml._events_logger.handlers):
        ml._events_logger.removeHandler(handler)
        handler.close()
    ml._configured_path = None


# --------------------------------------------------------------------------
# Event schema / storage (module-level, no vault.py involved)
# --------------------------------------------------------------------------


def test_record_writes_metadata_only_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("VAULT_MUTATION_LEDGER_DIR", str(tmp_path / "ledger"))
    ml.record(ml.MutationEvent(
        tool="vault_write", path="x.md", operation="create", result="success",
        old_hash="absent", new_hash="sha256:abc",
    ))
    events = ml.query_events(ledger_dir=tmp_path / "ledger")
    assert len(events) == 1
    event = events[0]
    assert set(event.keys()) == {
        "timestamp", "actor", "tool", "path", "operation", "destination",
        "old_hash", "new_hash", "result", "code", "correlation_id",
    }
    assert event["tool"] == "vault_write"
    assert event["operation"] == "create"
    assert event["result"] == "success"
    assert event["old_hash"] == "absent"
    assert event["new_hash"] == "sha256:abc"


def test_mode_off_skips_recording_entirely(tmp_path, monkeypatch):
    monkeypatch.setenv("VAULT_MUTATION_LEDGER_DIR", str(tmp_path / "ledger"))
    monkeypatch.setenv("VAULT_MUTATION_LEDGER_MODE", "off")
    ml.record(ml.MutationEvent(tool="vault_write", path="x.md", operation="create", result="success"))
    assert ml.query_events(ledger_dir=tmp_path / "ledger") == []
    assert not (tmp_path / "ledger").exists()


def test_unrecognised_mode_falls_back_to_on(monkeypatch):
    monkeypatch.setenv("VAULT_MUTATION_LEDGER_MODE", "bogus")
    assert ml._mode() == "on"


def test_query_events_filters_by_path_prefix_tool_and_result(tmp_path, monkeypatch):
    monkeypatch.setenv("VAULT_MUTATION_LEDGER_DIR", str(tmp_path / "ledger"))
    ml.record(ml.MutationEvent(tool="vault_write", path="notes/a.md", operation="create", result="success"))
    ml.record(ml.MutationEvent(tool="vault_delete", path="notes/b.md", operation="delete", result="reject", code="ValueError"))
    ml.record(ml.MutationEvent(tool="vault_write", path="other/c.md", operation="update", result="success"))

    assert len(ml.query_events(ledger_dir=tmp_path / "ledger")) == 3
    assert [e["path"] for e in ml.query_events(ledger_dir=tmp_path / "ledger", result="reject")] == ["notes/b.md"]
    assert [e["path"] for e in ml.query_events(ledger_dir=tmp_path / "ledger", tool="vault_write")] == ["other/c.md", "notes/a.md"]
    assert [e["path"] for e in ml.query_events(ledger_dir=tmp_path / "ledger", path_prefix="notes/")] == ["notes/b.md", "notes/a.md"]


def test_query_events_sorts_newest_first_and_respects_limit(tmp_path):
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    lines = [
        {"timestamp": "2026-08-17T10:00:00+00:00", "path": "a.md"},
        {"timestamp": "2026-08-17T12:00:00+00:00", "path": "b.md"},
        {"timestamp": "2026-08-17T11:00:00+00:00", "path": "c.md"},
    ]
    (ledger_dir / "mutations.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    events = ml.query_events(ledger_dir=ledger_dir, limit=2)
    assert [e["path"] for e in events] == ["b.md", "c.md"]


def test_query_events_skips_malformed_lines(tmp_path):
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "mutations.jsonl").write_text(
        '{"timestamp": "2026-08-17T10:00:00+00:00", "path": "a.md"}\n'
        "not json at all\n"
        "\n"
    )
    events = ml.query_events(ledger_dir=ledger_dir)
    assert [e["path"] for e in events] == ["a.md"]


def test_rotation_bounds_total_file_count(tmp_path, monkeypatch):
    ledger_dir = tmp_path / "ledger"
    monkeypatch.setenv("VAULT_MUTATION_LEDGER_DIR", str(ledger_dir))
    monkeypatch.setenv("VAULT_MUTATION_LEDGER_MAX_BYTES", "500")
    monkeypatch.setenv("VAULT_MUTATION_LEDGER_BACKUP_COUNT", "2")

    for i in range(200):
        ml.record(ml.MutationEvent(
            tool="vault_write", path=f"file-{i}.md", operation="update", result="success",
            old_hash="sha256:" + "a" * 64, new_hash="sha256:" + "b" * 64,
        ))

    files = list(ledger_dir.glob("mutations.jsonl*"))
    assert 1 <= len(files) <= 3  # primary + at most backupCount=2 rotated files
    assert any(f.name == "mutations.jsonl" for f in files)


def test_record_failure_never_raises_and_increments_failed_metric(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a regular file, not a directory")
    monkeypatch.setenv("VAULT_MUTATION_LEDGER_DIR", str(blocker / "ledger"))

    before = ml.health_metrics()["failed"]
    ml.record(ml.MutationEvent(tool="vault_write", path="x.md", operation="create", result="success"))
    after = ml.health_metrics()["failed"]

    assert after == before + 1


def test_emit_failure_after_handler_already_open_is_counted_failed_not_recorded(tmp_path, monkeypatch):
    """codex-review-phase2-write-integrity MEDIUM finding: stdlib logging
    handlers can swallow a write error internally (via the default
    handleError) so `_events_logger.info(line)` returns normally even though
    nothing was written -- record() used to unconditionally count that as
    "recorded". Force a real emit-time failure (handler already open, then
    its underlying stream is closed from under it) and confirm it's now
    counted as "failed", never "recorded".
    """
    monkeypatch.setenv("VAULT_MUTATION_LEDGER_DIR", str(tmp_path / "ledger"))

    # First call opens the handler successfully.
    ml.record(ml.MutationEvent(tool="vault_write", path="a.md", operation="create", result="success"))
    recorded_before = ml.health_metrics()["recorded"]
    failed_before = ml.health_metrics()["failed"]

    # Sabotage the already-open handler's stream so the NEXT emit fails --
    # this is exactly the "handler that follows logging's error-swallowing
    # behavior during emit" scenario from the review.
    handler = ml._events_logger.handlers[0]
    handler.stream.close()

    ml.record(ml.MutationEvent(tool="vault_write", path="b.md", operation="create", result="success"))

    metrics = ml.health_metrics()
    assert metrics["failed"] == failed_before + 1
    assert metrics["recorded"] == recorded_before  # NOT incremented for the failed emit

    # And the failed event must not silently appear as if it landed.
    events = ml.query_events(ledger_dir=tmp_path / "ledger")
    assert all(e.get("path") != "b.md" for e in events)

    # The handler's stream is now deliberately closed -- discard it directly
    # (rather than via handler.close(), which would itself raise trying to
    # flush a closed stream) so the shared autouse teardown fixture doesn't
    # choke on it.
    ml._events_logger.removeHandler(handler)
    ml._configured_path = None


def test_strict_handler_reraises_instead_of_swallowing():
    """Unit-level proof of the mechanism: _StrictRotatingFileHandler's
    handleError must re-raise rather than the stdlib default (print to
    stderr and continue)."""
    import logging

    handler = ml._StrictRotatingFileHandler.__new__(ml._StrictRotatingFileHandler)
    try:
        raise ValueError("boom")
    except ValueError:
        with pytest.raises(ValueError):
            handler.handleError(logging.makeLogRecord({}))


# --------------------------------------------------------------------------
# Integration through vault.py: write_file_atomic / move_path / delete_path
# --------------------------------------------------------------------------


def _ledger_dir(vault_dir):
    return vault_dir / ".mutation-ledger"


def test_create_event_for_new_file(vault_dir):
    write_file_atomic("brand-new.md", "hello", tool="vault_write")
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_write", path_prefix="brand-new.md")
    assert events[0]["operation"] == "create"
    assert events[0]["result"] == "success"
    assert events[0]["old_hash"] == "absent"
    assert events[0]["new_hash"] is not None


def test_update_event_for_existing_file(vault_dir):
    write_file_atomic("test-note.md", "changed body", tool="vault_write")
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_write", path_prefix="test-note.md")
    assert events[0]["operation"] == "update"
    assert events[0]["result"] == "success"
    assert events[0]["old_hash"] != events[0]["new_hash"]


def test_noop_write_with_stale_revision_still_succeeds_and_is_ledgered(vault_dir):
    content, meta = read_file("test-note.md")
    write_file_atomic("test-note.md", content, tool="vault_write", expected_revision="sha256:" + "0" * 64)
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_write", path_prefix="test-note.md")
    assert events[0]["result"] == "success"
    assert events[0]["old_hash"] == events[0]["new_hash"] == meta["revision"]


_VALID_OPERATIONS = {"create", "update", "delete", "move"}


def test_rejected_write_via_write_contract_gate_is_ledgered(vault_dir, monkeypatch):
    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE", "enforce")
    with pytest.raises(ValueError):
        write_file_atomic("bad:name.md", "hello", tool="vault_write")
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_write", path_prefix="bad:name.md")
    assert events[0]["result"] == "reject"
    assert events[0]["code"] == "write-contract-rejected"
    # codex-review-phase2-write-integrity MEDIUM: a rejected write must be
    # classified as the operation it would have been (create, since
    # bad:name.md doesn't exist), never the undeclared "write".
    assert events[0]["operation"] in _VALID_OPERATIONS
    assert events[0]["operation"] == "create"


def test_content_size_limit_rejection_is_ledgered(vault_dir, monkeypatch):
    monkeypatch.setattr("obsidian_vault_mcp.config.MAX_CONTENT_SIZE", 5)
    with pytest.raises(ValueError):
        write_file_atomic("too-big.md", "way more than five bytes", tool="vault_write")
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_write", path_prefix="too-big.md")
    assert events[0]["result"] == "reject"
    assert events[0]["code"] == "ValueError"
    assert events[0]["operation"] in _VALID_OPERATIONS


def test_conflict_on_write_is_ledgered(vault_dir):
    with pytest.raises(RevisionConflictError):
        write_file_atomic("test-note.md", "different content", tool="vault_write", expected_revision="sha256:" + "0" * 64)
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_write", path_prefix="test-note.md", result="conflict")
    assert events
    assert events[0]["code"] == "revision-conflict"
    # test-note.md already exists, so a conflicted guarded write on it is
    # classified "update", never the undeclared "write".
    assert events[0]["operation"] == "update"


def test_rejected_write_of_existing_file_is_classified_update_not_write(vault_dir, monkeypatch):
    """Same enum-consistency guarantee, but for a REJECT against a file that
    already exists (previously this and the create case above were
    indistinguishable -- both emitted the invalid "write")."""
    monkeypatch.setattr("obsidian_vault_mcp.config.MAX_CONTENT_SIZE", 1)
    with pytest.raises(ValueError):
        write_file_atomic("test-note.md", "hello", tool="vault_write")
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_write", path_prefix="test-note.md", result="reject")
    assert events
    assert events[0]["operation"] == "update"


def test_move_success_event(vault_dir):
    move_path("no-frontmatter.md", "moved.md")
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_move", result="success")
    assert events
    event = events[0]
    assert event["path"] == "no-frontmatter.md"
    assert event["destination"] == "moved.md"
    assert event["old_hash"] == event["new_hash"]


def test_move_reject_event_when_destination_exists(vault_dir):
    (vault_dir / "dest.md").write_text("already here")
    with pytest.raises(FileExistsError):
        move_path("test-note.md", "dest.md")
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_move", result="reject")
    assert events
    assert events[0]["code"] == "FileExistsError"


def test_move_conflict_event(vault_dir):
    with pytest.raises(RevisionConflictError):
        move_path("test-note.md", "elsewhere.md", expected_revision="sha256:" + "0" * 64)
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_move", result="conflict")
    assert events
    assert events[0]["code"] == "revision-conflict"


def test_delete_success_event(vault_dir):
    delete_path("no-frontmatter.md")
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_delete", result="success")
    assert events
    assert events[0]["path"] == "no-frontmatter.md"
    assert events[0]["new_hash"] is None


def test_delete_reject_event_for_nonempty_directory(vault_dir):
    with pytest.raises(ValueError):
        delete_path("subfolder")
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_delete", result="reject")
    assert events
    assert events[0]["code"] == "ValueError"


def test_delete_conflict_event(vault_dir):
    with pytest.raises(RevisionConflictError):
        delete_path("test-note.md", expected_revision="sha256:" + "0" * 64)
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_delete", result="conflict")
    assert events
    assert events[0]["code"] == "revision-conflict"


def test_correlation_id_and_actor_pass_through_when_supplied(vault_dir):
    write_file_atomic("test-note.md", "v2", tool="vault_write", correlation_id="corr-123", actor="agent-x")
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_write", path_prefix="test-note.md")
    assert events[0]["correlation_id"] == "corr-123"
    assert events[0]["actor"] == "agent-x"


def test_correlation_id_and_actor_default_to_none(vault_dir):
    write_file_atomic("test-note.md", "v3", tool="vault_write")
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_write", path_prefix="test-note.md")
    assert events[0]["correlation_id"] is None
    assert events[0]["actor"] is None


def test_ledger_never_contains_written_content_or_secrets(vault_dir):
    secret = "sk-super-secret-token-do-not-log-this"
    write_file_atomic("secret-note.md", f"top secret body containing {secret}", tool="vault_write")
    raw = (_ledger_dir(vault_dir) / "mutations.jsonl").read_text(encoding="utf-8")
    assert secret not in raw
    assert "top secret body" not in raw


def test_ledger_dir_excluded_from_vault_list(vault_dir):
    write_file_atomic("test-note.md", "updated", tool="vault_write")
    assert _ledger_dir(vault_dir).exists()
    result = json.loads(vault_list())
    names = {item["name"] for item in result["items"]}
    assert ".mutation-ledger" not in names


def test_mode_off_leaves_normal_write_and_delete_behaviour_unaffected(vault_dir, monkeypatch):
    monkeypatch.setenv("VAULT_MUTATION_LEDGER_MODE", "off")
    is_new, size = write_file_atomic("off-mode.md", "hello", tool="vault_write")
    assert (is_new, size) == (True, 5)
    assert delete_path("off-mode.md") is True
    assert not _ledger_dir(vault_dir).exists()


def test_vault_delete_tool_success_is_ledgered(vault_dir):
    result = json.loads(vault_delete("no-frontmatter.md", confirm=True))
    assert result["deleted"] is True
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_delete", result="success")
    assert events


def test_vault_move_tool_success_is_ledgered(vault_dir):
    result = json.loads(vault_move("no-frontmatter.md", "moved-via-tool.md"))
    assert result["moved"] is True
    events = ml.query_events(ledger_dir=_ledger_dir(vault_dir), tool="vault_move", result="success")
    assert events
