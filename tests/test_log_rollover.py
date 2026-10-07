"""Tests for scripts/log-rollover.py -- scheduled Write Rule 21 log rollover (dry-run by default)."""

import hashlib
import re
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


def test_failed_swap_leaves_original_intact_and_removes_the_copy(rollover, vault_dir, monkeypatch):
    p = _write(vault_dir, LOG, _log_text(bounded_files.LOG_ROLLOVER_BYTES))
    before = p.read_bytes()
    real = rollover.vault.write_file_atomic

    def fail_swap(rel, *a, **k):
        if rel == LOG:
            raise OSError("disk full")
        return real(rel, *a, **k)

    monkeypatch.setattr(rollover.vault, "write_file_atomic", fail_swap)
    results = rollover.run(mode="if-due", apply=True, today=TODAY)
    assert results[0]["action"] == "error"
    assert p.read_bytes() == before
    assert [f.name for f in p.parent.iterdir()] == [p.name]


def test_concurrent_append_during_apply_is_never_lost_and_path_never_missing(rollover, vault_dir, monkeypatch):
    """Witness: a vault_append landing between the copy and the swap."""
    from obsidian_vault_mcp.tools import write as write_tool

    p = _write(vault_dir, LOG, _log_text(bounded_files.LOG_ROLLOVER_BYTES))
    real = rollover.vault.write_file_atomic
    marker = "## 2026-10-08 — concurrent append marker"
    seen = {"calls": 0, "missing": False, "appended": False}

    def write_with_racing_append(rel, *a, **k):
        seen["calls"] += 1
        if not p.exists():
            seen["missing"] = True
        if rel != LOG and not seen["appended"]:
            # The copy is about to be written; an append lands on the live log right after it.
            out = real(rel, *a, **k)
            write_tool.vault_append(LOG, f"\n{marker}\n")
            seen["appended"] = True
            return out
        out = real(rel, *a, **k)
        if not p.exists():
            seen["missing"] = True
        return out

    monkeypatch.setattr(rollover.vault, "write_file_atomic", write_with_racing_append)
    results = rollover.run(mode="if-due", apply=True, today=TODAY)

    assert seen["appended"] and not seen["missing"]
    assert results[0]["action"] == "rolled"
    rolled = list(p.parent.glob("infrastructure-changelog-*.md"))
    assert len(rolled) == 1
    # The racing append landed in exactly one place: the old volume (retry re-planned against it).
    assert rolled[0].read_text().count(marker) + p.read_text().count(marker) == 1
    assert rolled[0].read_text().count(marker) == 1
    assert p.read_text().startswith("---\n") and "continued_from:" in p.read_text()


def test_append_after_apply_lands_in_new_volume(rollover, vault_dir):
    from obsidian_vault_mcp.tools import write as write_tool

    p = _write(vault_dir, LOG, _log_text(bounded_files.LOG_ROLLOVER_BYTES))
    rollover.run(mode="if-due", apply=True, today=TODAY)
    write_tool.vault_append(LOG, "\n## 2026-11-02 — after the roll\n")
    assert "after the roll" in p.read_text()
    rolled = next(p.parent.glob("infrastructure-changelog-2026-*.md"))
    assert "after the roll" not in rolled.read_text()


def test_never_touches_hot_md_and_skips_missing_logs(rollover, vault_dir):
    hot = _write(vault_dir, "BS 2nd Brain/Alcove/Infrastructure/hot.md", "x" * 900_000)
    log2 = _write(vault_dir, LOG2, "# Log\n\n## 2026-10-01 — entry\n")
    hot_before, log2_before = hot.read_bytes(), log2.read_bytes()

    results = rollover.run(mode="if-due", apply=True, today=TODAY)

    assert {r["path"] for r in results} == {LOG2}
    assert hot.read_bytes() == hot_before
    assert log2.read_bytes() == log2_before


# --- new volume matches the 2026-10-07 hand roll ------------------------------

INFRA = "BS 2nd Brain/Alcove/Infrastructure"
PREV_VOLUME = f"{INFRA}/infrastructure-changelog-2026-09-08-to-2026-10-06"
OLDEST_VOLUME = f"{INFRA}/infrastructure-changelog-through-2026-09-07"
LIVE_NOTE = (
    "> Reverse-chronological: newest entry directly below this note. Insert with `vault_str_replace` "
    "anchored on the current newest heading; never `vault_write` or `vault_append` this file "
    "(Write Rule 13). Rolled over monthly or at 500KB, whichever comes first (Write Rule 21)."
)
# Header of the live infrastructure-changelog.md as rolled by hand on 2026-10-07 (entries trimmed).
LIVE_CHANGELOG = f"""---
continued_from: {PREV_VOLUME}.md
created: '2026-10-07'
read_policy: section-only
type: infrastructure-changelog
updated: '2026-10-07'
last_edited_by: Claude (Anthropic)
last_edited_via: Claude.ai with BS Brain MCP
last_edit_note: '2026-10-07: monthly rollover (Write Rule 21). Previous volume moved byte-for-byte; this file starts empty apart from the rollover entry.'
---

# Infrastructure changelog

{LIVE_NOTE} Earlier volumes: [[{PREV_VOLUME}]], [[{OLDEST_VOLUME}]].

## 2026-10-07 — G-fixes deploy v3: release a61c637
**Status:** executed

## 2026-10-07 — Brain housekeeping: hot.md compiled truth, changelog rollover, Write Rule 21
**Status:** executed
"""


def _header(text):
    """(frontmatter dict, preamble text) of a log volume: everything before its first dated entry."""
    meta, body = bounded_files.split_frontmatter(text)
    head = body[: bounded_files.DIARY_HEADING_RE.search(body).start()]
    return meta, head


def test_new_volume_header_matches_the_live_hand_roll(rollover, vault_dir):
    live = _write(vault_dir, LOG, LIVE_CHANGELOG)
    _write(vault_dir, f"{PREV_VOLUME}.md", "---\ncontinued_from: " + OLDEST_VOLUME + ".md\n---\n# Infrastructure changelog\n")
    _write(vault_dir, f"{OLDEST_VOLUME}.md", "# Infrastructure changelog\n")

    results = rollover.run(mode="monthly", apply=True, today=date(2026, 11, 1))
    assert results[0]["action"] == "rolled"
    rolled_target = f"{INFRA}/infrastructure-changelog-2026-10-07-to-2026-10-07"

    live_meta, live_head = _header(LIVE_CHANGELOG)
    new_meta, new_head = _header(live.read_text())

    # Frontmatter: the same keys the hand roll set, pointing at the volume just rolled.
    assert new_meta["continued_from"] == f"{rolled_target}.md"
    assert new_meta["read_policy"] == live_meta["read_policy"]
    assert new_meta["type"] == live_meta["type"]
    assert {"created", "updated", "last_edit_note"} <= set(new_meta)

    # Body header: identical writer note (Rule 13 guidance kept), and *all* earlier volumes linked.
    def split_note(head):
        note, _, links = head.partition(" Earlier volumes:")
        return note.strip(), re.findall(r"\[\[([^\]]+)\]\]", links)

    live_note, live_links = split_note(live_head)
    new_note, new_links = split_note(new_head)
    assert "never `vault_write` or `vault_append` this file (Write Rule 13)" in new_note
    assert new_note.replace("# Infrastructure changelog", "").strip() == live_note.replace("# Infrastructure changelog", "").strip()
    assert new_links == [rolled_target, PREV_VOLUME, OLDEST_VOLUME]
    assert new_links[1:] == live_links


def test_volume_without_a_writer_note_gets_the_changelog_default_and_full_chain(rollover, vault_dir):
    # The 2026-09-08 volume had no note at all; its continued_from chain still yields every earlier volume.
    text = (
        f"---\ncontinued_from: {OLDEST_VOLUME}.md\nread_policy: section-only\ntype: infrastructure-changelog\n---\n"
        "# Infrastructure changelog\n\n## 2026-10-06 — Newest\nbody\n\n## 2026-09-08 — Oldest\nbody\n"
    )
    live = _write(vault_dir, LOG, text)
    _write(vault_dir, f"{OLDEST_VOLUME}.md", "# Infrastructure changelog\n")

    rollover.run(mode="monthly", apply=True, today=date(2026, 10, 7))

    new = live.read_text()
    assert "(Write Rule 13)" in new and "never `vault_write` or `vault_append` this file" in new
    links = re.findall(r"\[\[([^\]]+)\]\]", new.split("## 2026-10-07")[0])
    assert links == [f"{INFRA}/infrastructure-changelog-2026-09-08-to-2026-10-06", OLDEST_VOLUME]


def test_other_logs_get_no_invented_writer_note(rollover, vault_dir):
    live = _write(vault_dir, LOG2, "---\ntype: log\n---\n# Log\n\n## 2026-09-01 — entry\nbody\n")
    rollover.run(mode="monthly", apply=True, today=TODAY)
    head = live.read_text().split("## 2026-11-01")[0]
    assert "Write Rule 13" not in head
    assert "Earlier volumes: [[BS 2nd Brain/_log-2026-09-01-to-2026-09-01]]." in head
