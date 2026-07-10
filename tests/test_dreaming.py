"""Tests for scripts/dreaming.py -- the nightly report-only vault maintenance cycle."""

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "dreaming.py"


@pytest.fixture(scope="module")
def dreaming():
    spec = importlib.util.spec_from_file_location("dreaming", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def vault(tmp_path):
    """A vault with a broken link, a hot.md over budget, an archive candidate,
    and two same-titled notes."""
    (tmp_path / "good-note.md").write_text(
        "---\ntype: note\n---\n\n# Good Note\n\nLinks to [[good-note-2]].\n"
    )
    (tmp_path / "good-note-2.md").write_text("# Good Note 2\n\nSome content.\n")
    (tmp_path / "broken-link.md").write_text(
        "# Broken Link Note\n\nSee [[does-not-exist]] for details.\n"
    )

    hot = tmp_path / "hot.md"
    hot.write_text("x" * 3000)

    old_prompt = tmp_path / "old-prompt.md"
    old_prompt.write_text("---\ntype: cc-prompt\nstatus: done\n---\n\n# Old Prompt\n\nDone.\n")
    import os
    old_time = datetime.now(timezone.utc).timestamp() - (40 * 86400)
    os.utime(old_prompt, (old_time, old_time))

    (tmp_path / "dup-a.md").write_text("# Same Title\n\nFirst copy.\n")
    (tmp_path / "dup-b.md").write_text("# Same Title\n\nSecond copy.\n")

    excluded = tmp_path / ".trash"
    excluded.mkdir()
    (excluded / "ignored.md").write_text("# Should be ignored\n")

    return tmp_path


def test_list_md_files_excludes_dirs(dreaming, vault):
    files = dreaming.list_md_files(vault)
    assert ".trash/ignored.md" not in files
    assert "good-note.md" in files
    assert all(f.endswith(".md") for f in files)


def test_list_md_files_excludes_prior_reports(dreaming, tmp_path):
    (tmp_path / "note.md").write_text("# Note\n")
    reports = tmp_path / "_Reports" / "dreaming"
    reports.mkdir(parents=True)
    (reports / "2026-07-09.md").write_text("- [ ] Fix broken link: `[[does-not-exist]]`\n")

    other_reports = tmp_path / "_Reports" / "other"
    other_reports.mkdir(parents=True)
    (other_reports / "keep.md").write_text("# Keep\n")

    bs_reports = tmp_path / "BS 2nd Brain" / "Alcove" / "Infrastructure" / "dreaming-reports"
    bs_reports.mkdir(parents=True)
    (bs_reports / "2026-07-09.md").write_text("- [ ] Fix broken link: `[[does-not-exist]]`\n")

    files = dreaming.list_md_files(tmp_path)
    assert "note.md" in files
    assert "_Reports/other/keep.md" in files
    assert not any("dreaming" in f for f in files)


def test_broken_wikilinks_detects_missing_target(dreaming, vault):
    md_files = dreaming.list_md_files(vault)
    broken = dreaming.pass_broken_wikilinks(vault, md_files)
    assert {"file": "broken-link.md", "link": "does-not-exist"} in broken
    assert not any(b["file"] == "good-note.md" for b in broken)


def test_archive_candidates_flags_stale_completed_prompt(dreaming, vault):
    md_files = dreaming.list_md_files(vault)
    now = datetime.now(timezone.utc)
    candidates = dreaming.pass_archive_candidates(vault, md_files, now)
    paths = [c["path"] for c in candidates]
    assert "old-prompt.md" in paths
    assert "good-note.md" not in paths


def test_hot_md_budget_flags_oversized_file(dreaming, vault):
    md_files = dreaming.list_md_files(vault)
    flagged = dreaming.pass_hot_md_budget(vault, md_files)
    assert len(flagged) == 1
    assert flagged[0]["path"] == "hot.md"
    assert flagged[0]["chars"] == 3000


def test_hot_md_budget_ignores_small_file(dreaming, tmp_path):
    (tmp_path / "hot.md").write_text("short")
    flagged = dreaming.pass_hot_md_budget(tmp_path, dreaming.list_md_files(tmp_path))
    assert flagged == []


def test_near_duplicates_title_match(dreaming, vault):
    md_files = dreaming.list_md_files(vault)
    result = dreaming.pass_near_duplicates(vault, md_files)
    title_matches = result["title_matches"]
    assert any(
        set(tm["files"]) == {"dup-a.md", "dup-b.md"} for tm in title_matches
    )


def test_report_path_bs_brain_vs_other(dreaming, tmp_path):
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    bs_path = dreaming.report_path_for(tmp_path, "bs-brain", now)
    assert str(bs_path).endswith("BS 2nd Brain/Alcove/Infrastructure/dreaming-reports/2026-07-10.md")

    cb_path = dreaming.report_path_for(tmp_path, "cb-brain", now)
    assert str(cb_path).endswith("_Reports/dreaming/2026-07-10.md")


def test_contradiction_lint_skips_non_sunday(dreaming, tmp_path):
    monday = datetime(2026, 7, 6, tzinfo=timezone.utc)
    assert dreaming.pass_contradiction_lint_sunday(tmp_path, "bs-brain", monday) is None


def test_contradiction_lint_skips_non_bs_brain_on_sunday(dreaming, tmp_path):
    sunday = datetime(2026, 7, 12, tzinfo=timezone.utc)
    assert dreaming.pass_contradiction_lint_sunday(tmp_path, "cb-brain", sunday) is None


def test_contradiction_lint_missing_files_reports_skip(dreaming, tmp_path):
    sunday = datetime(2026, 7, 12, tzinfo=timezone.utc)
    result = dreaming.pass_contradiction_lint_sunday(tmp_path, "bs-brain", sunday)
    assert result["status"] == "skipped"


def test_build_report_has_what_this_means_and_proposed_actions(dreaming):
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    report = dreaming.build_report(
        "bs-brain",
        now,
        {"status": "skipped", "reason": "not available in test"},
        [],
        [],
        [],
        {"title_matches": [], "embedding_matches": []},
        None,
    )
    assert "## What this means" in report
    assert "## Proposed actions" in report
    assert "Nothing to action tonight" in report


def test_run_writes_report_and_entities_json_leaves_content_untouched(dreaming, vault, monkeypatch):
    monkeypatch.setattr(dreaming, "VAULT_PATH", vault)
    monkeypatch.setattr(dreaming, "VAULT_NAME", "cb-brain")
    monkeypatch.setattr(dreaming.ss, "SEMANTIC_AVAILABLE", False)

    before = {p: p.stat().st_mtime for p in vault.rglob("*.md")}

    out_path = dreaming.run()

    assert out_path.exists()
    assert out_path.parent == vault / "_Reports" / "dreaming"

    entities_path = vault / "_entities.json"
    assert entities_path.exists()

    after_md_files = set(vault.rglob("*.md")) - {out_path}
    for p in after_md_files:
        assert p.stat().st_mtime == before[p]


# --- Entity index -----------------------------------------------------------

def test_generate_aliases_single_person(dreaming):
    assert dreaming.generate_aliases("Asimus, Angie") == ["Angie Asimus"]


def test_generate_aliases_couple_both_surnames(dreaming):
    assert dreaming.generate_aliases("Baader, Benjamin & Jacquet, Aurelie") == [
        "Benjamin Baader",
        "Aurelie Jacquet",
    ]


def test_generate_aliases_couple_shared_surname(dreaming):
    assert dreaming.generate_aliases("Duff, Scott & Tracey") == ["Scott Duff", "Tracey Duff"]


def test_generate_aliases_no_comma_returns_empty(dreaming):
    assert dreaming.generate_aliases("Sajju Shrestha") == []


@pytest.fixture
def entity_vault(tmp_path):
    """BS-Brain-shaped vault: Clients/ folder entities, a couple file, an
    aliased entity, a frontmatter-type-only entity outside any entity folder,
    and mentions to backlink against."""
    clients = tmp_path / "BS 2nd Brain" / "Alcove" / "Clients"
    clients.mkdir(parents=True)
    (clients / "McGrath, Danny.md").write_text(
        "---\ntype: client\n---\n\n# McGrath, Danny\n\nRefinance in progress.\n"
    )
    (clients / "McGrath, Michael & McGrath, Kim.md").write_text(
        "---\ntype: client\n---\n\n# McGrath, Michael & McGrath, Kim\n\nSettled.\n"
    )
    (clients / "Robson, Lloyd & McGrath, Rebecca.md").write_text(
        "---\ntype: client\naliases: [\"The Robsons\"]\n---\n\n# Robson, Lloyd & McGrath, Rebecca\n"
    )

    other = tmp_path / "Notes"
    other.mkdir()
    (other / "meeting-note.md").write_text(
        "# Meeting\n\nCalled [[McGrath, Danny]] about the refinance.\n"
        "Also spoke with Michael McGrath by phone.\n"
    )
    (other / "reference-note.md").write_text(
        "---\ntype: reference\n---\n\n# Alcove Partners\n\nGeneral reference note.\n"
    )

    return tmp_path


def test_entity_candidates_includes_folder_and_frontmatter_type(dreaming, entity_vault):
    md_files = dreaming.list_md_files(entity_vault)
    candidates = dreaming._entity_candidates(entity_vault, "bs-brain", md_files)
    assert "BS 2nd Brain/Alcove/Clients/McGrath, Danny.md" in candidates
    assert "Notes/reference-note.md" in candidates
    assert "Notes/meeting-note.md" not in candidates


def test_pass_entity_index_mcgrath_disambiguation_and_backlinks(dreaming, entity_vault):
    md_files = dreaming.list_md_files(entity_vault)
    entities = dreaming.pass_entity_index(entity_vault, "bs-brain", md_files)

    mcgrath_matches = [e for e in entities if "mcgrath" in e["name"].lower()]
    assert len(mcgrath_matches) == 3

    danny = next(e for e in entities if e["name"] == "McGrath, Danny")
    assert danny["aliases"] == ["Danny McGrath"]
    backlink_paths = {b["path"] for b in danny["backlinks"]}
    assert "Notes/meeting-note.md" in backlink_paths

    couple = next(e for e in entities if e["name"] == "McGrath, Michael & McGrath, Kim")
    assert couple["aliases"] == ["Michael McGrath", "Kim McGrath"]
    couple_backlink_paths = {b["path"] for b in couple["backlinks"]}
    assert "Notes/meeting-note.md" in couple_backlink_paths

    robson = next(e for e in entities if e["name"] == "Robson, Lloyd & McGrath, Rebecca")
    assert "The Robsons" in robson["aliases"]


def test_write_entities_json_schema(dreaming, entity_vault):
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    entities = dreaming.pass_entity_index(entity_vault, "bs-brain", dreaming.list_md_files(entity_vault))
    out_path = dreaming.write_entities_json(entity_vault, "bs-brain", now, entities)

    assert out_path == entity_vault / "_entities.json"
    payload = json.loads(out_path.read_text())
    assert payload["vault"] == "bs-brain"
    assert payload["entity_count"] == len(entities)
    assert payload["entity_count"] >= 4
