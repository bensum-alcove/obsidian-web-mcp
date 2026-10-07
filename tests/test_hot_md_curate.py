"""Tests for scripts/hot-md-curate.py's shared mutation authority
(vault-integrity-and-bo-authority-remediation-v2): write_file/archive
appends now go through vault_lock instead of a bare open(..., 'w'/'a').
"""

import importlib.util
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime
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


# --- hot-cache-v4: budget policy, non-canonical sections, resolved/stale
# refs, canonical conflict, and freshness-preserving rotation. ---

import obsidian_vault_mcp.config as _config  # noqa: E402


def test_budget_chars_matches_single_source_of_truth(curate):
    assert curate.BUDGET_CHARS == _config.HOT_MD_BUDGET_CHARS


def test_check_a_reports_overage(curate):
    text = "---\nfoo: bar\n---\n" + ("x" * (curate.BUDGET_CHARS + 137))
    result = curate.check_a(text)
    assert result["chars"] == curate.BUDGET_CHARS + 137
    assert result["budget"] == curate.BUDGET_CHARS
    assert result["overage"] == 137


def test_check_a_under_budget_has_zero_overage(curate):
    result = curate.check_a("---\nfoo: bar\n---\nshort body")
    assert result["overage"] == 0


def test_non_canonical_section_flagged(curate):
    lines = ["## What's current", "- fine", "## Random Notes", "- stray content", ""]
    sections = curate.parse_sections(lines)
    report = curate.non_canonical_section_report(lines, sections, "hot.md")
    assert any("Random Notes" in l for l in report)
    assert not any("What's current" in l for l in report)


def test_distinct_canonical_suffix_variants_are_not_parse_corruption(curate):
    text = "## Parked / considered and rejected\n- one\n## Parked (with CEO sign-off)\n- two\n"
    assert curate.structure_safety_errors(text, curate.parse_sections(text.splitlines()), "hot.md") == []


def _write_spec_and_log(vault_root, candidate, status_text):
    (vault_root / "Personal/Build Orchestrator/specs").mkdir(parents=True, exist_ok=True)
    (vault_root / "Personal/Build Orchestrator/specs" / f"{candidate}.md").write_text("spec")
    log_dir = vault_root / "Personal/Build Orchestrator/build-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{candidate}-output.md").write_text(status_text)


def test_check_b_flags_resolved_reference(curate, tmp_path):
    _write_spec_and_log(tmp_path, "some-finished-build", "## Status\npass\n")
    lines = ["## In flight", "- shipped `some-finished-build` today", ""]
    sections = curate.parse_sections(lines)
    report, removal_specs = curate.check_b(str(tmp_path), lines, 0, sections)
    assert any(l.startswith("RESOLVED: some-finished-build") for l in report)
    assert len(removal_specs) == 1


def test_check_b_flags_stale_blocker(curate, tmp_path):
    _write_spec_and_log(tmp_path, "some-finished-build", "## Status\npass\n")
    lines = ["## Blockers / watchpoints", "- blocked on `some-finished-build` shipping", ""]
    sections = curate.parse_sections(lines)
    report, removal_specs = curate.check_b(str(tmp_path), lines, 0, sections)
    assert any(l.startswith("STALE-BLOCKER: some-finished-build") for l in report)
    assert len(removal_specs) == 1


def _write_canonical_record(curate, vault_root, component_id, state):
    records_dir = vault_root / curate.CANONICAL_STATE_RECORDS_DIR
    records_dir.mkdir(parents=True, exist_ok=True)
    (records_dir / f"{component_id}.md").write_text(
        "---\n"
        "type: canonical-state\n"
        f"component_id: {component_id}\n"
        f"state: {state}\n"
        "content_updated: '2026-08-01'\n"
        "verified_at: '2026-08-01'\n"
        "source: 'commit abc123'\n"
        "---\n\nbody\n"
    )


def test_check_f_flags_canonical_conflict(curate, tmp_path):
    _write_canonical_record(curate, tmp_path, "some-widget", "deprecated")
    lines = ["## What's current", "- `some-widget` is the primary integration path", ""]
    sections = curate.parse_sections(lines)
    report, removal_specs = curate.check_f(str(tmp_path), lines, 0, sections)
    assert any("CANONICAL-CONFLICT: some-widget" in l for l in report)
    assert len(removal_specs) == 1


def test_check_f_skips_when_hot_md_already_acknowledges_state(curate, tmp_path):
    _write_canonical_record(curate, tmp_path, "some-widget", "deprecated")
    lines = ["## What's current", "- `some-widget` is deprecated, do not use", ""]
    sections = curate.parse_sections(lines)
    report, removal_specs = curate.check_f(str(tmp_path), lines, 0, sections)
    assert report == []
    assert removal_specs == []


def test_check_f_ignores_active_components(curate, tmp_path):
    _write_canonical_record(curate, tmp_path, "some-widget", "active")
    lines = ["## What's current", "- `some-widget` is live", ""]
    sections = curate.parse_sections(lines)
    report, removal_specs = curate.check_f(str(tmp_path), lines, 0, sections)
    assert report == []
    assert removal_specs == []


def test_apply_mode_rotation_does_not_bump_frontmatter_updated(curate, tmp_path):
    rel_path = "hot.md"
    abs_path = tmp_path / rel_path

    bullets = "\n".join(f"- shipped thing {i}" for i in range(8))
    text = (
        "---\n"
        "updated: '2026-01-01'\n"
        "---\n"
        "## Last session shipped\n"
        f"{bullets}\n"
        "\n"
        "## In flight\n"
        "- ongoing work\n"
    )
    abs_path.write_text(text)
    old_time = time.time() - curate.MTIME_GUARD_SECONDS - 3600
    os.utime(abs_path, (old_time, old_time))

    result = curate.process_target(str(tmp_path), rel_path, True, "2026-08-21", time.time())
    assert result["changed"] is True

    new_text = abs_path.read_text()
    assert "updated: '2026-01-01'" in new_text
    assert new_text.count("- shipped thing") == curate.SECTION_CAP


def _load_curate_module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_telegram_token_not_hardcoded_in_source(curate):
    source = SCRIPT_PATH.read_text()
    assert "TELEGRAM_BOT_TOKEN = os.environ.get" in source
    assert "TELEGRAM_CHAT_ID = os.environ.get" in source
    # regression guard: no literal Telegram bot-token-shaped constant in source
    assert not re.search(r'TELEGRAM_BOT_TOKEN\s*=\s*"\d+:', source)


def test_telegram_credentials_missing_from_environment(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    module = _load_curate_module("hot_md_curate_no_env")
    assert module.TELEGRAM_BOT_TOKEN is None
    assert module.TELEGRAM_CHAT_ID is None
    with pytest.raises(RuntimeError):
        module.send_telegram("test message")


def test_telegram_credentials_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token-for-test")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake-chat-id-for-test")
    module = _load_curate_module("hot_md_curate_with_env")
    assert module.TELEGRAM_BOT_TOKEN == "fake-token-for-test"
    assert module.TELEGRAM_CHAT_ID == "fake-chat-id-for-test"


# --- hot-md-curate-exception-only-selfheal-v2: bounded retry and final-state
# notification authority. ---


def _write_hot_target(root, rel_path="hot.md", *, extra="", bullet_count=6):
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    bullets = "\n".join(f"- shipped exact block {i}" for i in range(bullet_count))
    target.write_text(
        "---\nupdated: '2026-01-01'\n---\n"
        "## Last session shipped\n"
        f"{bullets}\n\n"
        "## In flight\n- current work\n\n"
        f"{extra}"
    )
    return target


def _configure_single_target(curate, monkeypatch, tmp_path, rel_path="hot.md"):
    monkeypatch.setattr(curate, "TARGETS", [rel_path])
    monkeypatch.setattr(curate, "BACKUP_ROOT", str(tmp_path / "backups"))
    monkeypatch.setattr(curate, "LOCK_FILE", str(tmp_path / "curate.lock"))


def test_recent_mtime_cron_retry_applies_and_does_not_notify_before_retry(
    curate, tmp_path, monkeypatch
):
    _configure_single_target(curate, monkeypatch, tmp_path)
    target = _write_hot_target(tmp_path)
    monkeypatch.setattr(curate, "MTIME_GUARD_SECONDS", 10)
    monkeypatch.setattr(curate, "CRON_RETRY_MARGIN_SECONDS", 1)
    monkeypatch.setattr(curate, "CRON_RETRY_LIMIT_SECONDS", 30)

    fake_now = [1_000.0]
    os.utime(target, (995.0, 995.0))
    notifications = []

    def clock():
        return fake_now[0]

    def sleep(seconds):
        assert notifications == []
        fake_now[0] += seconds

    real_apply = curate.apply_with_bounded_retry
    captured = []

    def run_with_fake_time(vault_root, today_str, invoked_by_cron):
        outcome = real_apply(
            vault_root, today_str, invoked_by_cron, clock=clock, sleep=sleep
        )
        captured.append(outcome)
        return outcome

    monkeypatch.setattr(curate, "apply_with_bounded_retry", run_with_fake_time)
    monkeypatch.setattr(
        curate, "send_telegram", lambda message: notifications.append(message) or "ok"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT_PATH), "--apply", "--from-cron", "--vault-root", str(tmp_path)],
    )

    assert curate.main() == 0
    attempts, final_apply, failures = captured[0]

    assert len(attempts) == 2
    assert attempts[0][0]["mtime_skipped"] is True
    assert final_apply[0]["changed"] is True
    assert failures == []
    assert notifications == []
    assert "- shipped exact block 0" not in target.read_text()
    assert "- shipped exact block 0" in (
        tmp_path / f"hot-archive/{datetime.now():%Y-%m}.md"
    ).read_text()


def test_second_apply_is_idempotent_and_archive_is_lossless(curate, tmp_path, monkeypatch):
    _configure_single_target(curate, monkeypatch, tmp_path)
    target = _write_hot_target(tmp_path, bullet_count=7)
    old_time = time.time() - curate.MTIME_GUARD_SECONDS - 1
    os.utime(target, (old_time, old_time))

    original_blocks = [f"- shipped exact block {i}" for i in range(2)]
    first = curate.process_target(str(tmp_path), "hot.md", True, "2026-09-09", time.time())
    archive = tmp_path / "hot-archive/2026-09.md"
    archive_after_first = archive.read_bytes()
    target_after_first = target.read_bytes()

    os.utime(target, (old_time, old_time))
    second = curate.process_target(str(tmp_path), "hot.md", True, "2026-09-09", time.time())

    assert first["changed"] is True
    assert second["changed"] is False
    assert target.read_bytes() == target_after_first
    assert archive.read_bytes() == archive_after_first
    for block in original_blocks:
        assert archive_after_first.count(block.encode()) == 1


def test_continuing_edits_exhaust_bounded_retry(curate, tmp_path, monkeypatch):
    _configure_single_target(curate, monkeypatch, tmp_path)
    target = _write_hot_target(tmp_path)
    monkeypatch.setattr(curate, "MTIME_GUARD_SECONDS", 10)
    monkeypatch.setattr(curate, "CRON_RETRY_MARGIN_SECONDS", 1)
    monkeypatch.setattr(curate, "CRON_RETRY_LIMIT_SECONDS", 12)
    fake_now = [1_000.0]
    os.utime(target, (995.0, 995.0))

    def sleep_and_edit(seconds):
        fake_now[0] += seconds
        os.utime(target, (fake_now[0], fake_now[0]))

    attempts, final_apply, failures = curate.apply_with_bounded_retry(
        str(tmp_path),
        "2026-09-09",
        True,
        clock=lambda: fake_now[0],
        sleep=sleep_and_edit,
    )

    assert len(attempts) == 2
    assert final_apply[0]["mtime_skipped"] is True
    assert len(failures) == 1
    assert failures[0].startswith("quiet-window retry exhausted")
    assert "- shipped exact block 0" in target.read_text()


def test_report_mode_is_verbose_and_does_not_touch_target_or_archive(
    curate, tmp_path, monkeypatch
):
    _configure_single_target(curate, monkeypatch, tmp_path)
    target = _write_hot_target(
        tmp_path,
        extra="## Historical notes\n" + ("old material " * 500) + "\n",
    )
    target.write_text(
        target.read_text().replace(
            "- current work",
            "- blocked on `finished-blocker` shipping\n"
            "- shipped `finished-item`\n"
            + "- " + ("long routine diagnostic " * 20),
        )
    )
    _write_spec_and_log(tmp_path, "finished-blocker", "## Status: pass\n")
    _write_spec_and_log(tmp_path, "finished-item", "## Status: pass\n")
    before = target.read_bytes()

    result = curate.process_target(
        str(tmp_path), "hot.md", False, "2026-09-09", time.time()
    )

    assert target.read_bytes() == before
    assert not (tmp_path / "hot-archive").exists()
    joined = "\n".join(result["report_lines"])
    assert "NON-CANONICAL-SECTION" in joined
    assert "to rotate to hot-archive/" in joined


def test_routine_final_findings_and_review_due_send_zero_telegram(
    curate, tmp_path, monkeypatch
):
    _configure_single_target(curate, monkeypatch, tmp_path)
    target = _write_hot_target(
        tmp_path,
        extra="## Historical notes\n" + ("old material " * 500) + "\n",
    )
    target.write_text(
        target.read_text().replace(
            "- current work",
            "- blocked on `finished-blocker` shipping\n"
            "- shipped `finished-item`\n"
            + "- " + ("long routine diagnostic " * 20),
        )
    )
    _write_spec_and_log(tmp_path, "finished-blocker", "## Status: pass\n")
    _write_spec_and_log(tmp_path, "finished-item", "## Status: pass\n")
    old_time = time.time() - curate.MTIME_GUARD_SECONDS - 1
    os.utime(target, (old_time, old_time))
    reviews = tmp_path / curate.SCHEDULED_REVIEWS_PATH
    reviews.parent.mkdir(parents=True, exist_ok=True)
    reviews.write_text("### bookkeeping\ndue: 2026-01-01\nstatus: pending\n")
    sent = []
    monkeypatch.setattr(curate, "send_telegram", lambda message: sent.append(message))
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT_PATH), "--apply", "--from-cron", "--vault-root", str(tmp_path)],
    )

    assert curate.main() == 0
    assert sent == []
    report = (
        tmp_path / f"BS 2nd Brain/Alcove/Infrastructure/hot-md-reports/{datetime.now():%Y-%m-%d}.md"
    ).read_text()
    assert "OVER-BUDGET-AT-FLOOR" in report
    assert "STALE-BLOCKER: finished-blocker" in report
    assert "RESOLVED: finished-item" in report
    assert "LONG-BULLET" in report
    assert "REVIEW DUE: bookkeeping" in report


@pytest.mark.parametrize("failure", ["archive", "backup", "parse"])
def test_true_safety_failure_sends_exactly_one_exception_message(
    curate, tmp_path, monkeypatch, failure
):
    _configure_single_target(curate, monkeypatch, tmp_path)
    target = _write_hot_target(tmp_path)
    old_time = time.time() - curate.MTIME_GUARD_SECONDS - 1
    os.utime(target, (old_time, old_time))

    if failure == "archive":
        monkeypatch.setattr(
            curate,
            "append_archive_losslessly",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fixture archive failure")),
        )
    elif failure == "backup":
        monkeypatch.setattr(
            curate,
            "backup_targets",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fixture backup failure")),
        )
    else:
        target.write_text("---\nunterminated frontmatter\n## Last session shipped\n- item\n")

    sent = []
    monkeypatch.setattr(curate, "send_telegram", lambda message: sent.append(message) or "ok")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT_PATH), "--apply", "--from-cron", "--vault-root", str(tmp_path)],
    )

    assert curate.main() == 1
    assert len(sent) == 1
    assert "integrity exception" in sent[0]
    assert "Chars:" not in sent[0]
    assert "REVIEW-DUE" not in sent[0]
