"""Tests for scripts/dreaming.py -- the nightly report-only vault maintenance cycle."""

import importlib.util
import json
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
    result = dreaming.pass_broken_wikilinks(vault, md_files)
    broken = result["broken"]
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
        {"broken": [], "suppressed_count": 0},
        [],
        [],
        {"title_matches": [], "embedding_matches": [], "suspect_lines": []},
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


# --- Denoise: code-fence-aware extraction + placeholder allowlist -----------

def test_broken_wikilinks_ignores_bash_test_syntax_in_fence(dreaming, tmp_path):
    (tmp_path / "ha-diagnose.md").write_text(
        "# HA Diagnose\n\n```bash\nif [[ \"$HA_CODE\" == \"000\" ]]; then\n  echo ok\nfi\n```\n"
    )
    md_files = dreaming.list_md_files(tmp_path)
    result = dreaming.pass_broken_wikilinks(tmp_path, md_files)
    assert result["broken"] == []
    assert result["suppressed_count"] == 0  # stripped before extraction, not suppressed


def test_broken_wikilinks_ignores_tilde_fence_and_inline_code(dreaming, tmp_path):
    (tmp_path / "note.md").write_text(
        "# Note\n\n~~~\n[[ \"$X\" == \"1\" ]]\n~~~\n\nSee `[[inline-code-link]]` here.\n"
    )
    md_files = dreaming.list_md_files(tmp_path)
    result = dreaming.pass_broken_wikilinks(tmp_path, md_files)
    assert result["broken"] == []


def test_broken_wikilinks_suppresses_placeholders_and_identifiers(dreaming, tmp_path):
    (tmp_path / "schema.md").write_text(
        "# Schema\n\n"
        "Example: [[wikilink]] or [[wikilinks]] or [[target]] or [[entity]] or [[note name]].\n"
        "Full example: [[BS 2nd Brain/path/to/file]].\n"
        "Internal id: [[project_shadow_execution_gap]].\n"
        "Elided: [[Alcove/...]].\n"
        "Bare dir: [[Alcove/Systems/]].\n"
    )
    md_files = dreaming.list_md_files(tmp_path)
    result = dreaming.pass_broken_wikilinks(tmp_path, md_files)
    assert result["broken"] == []
    assert result["suppressed_count"] == 9


def test_broken_wikilinks_still_reports_genuine_broken_link(dreaming, tmp_path):
    (tmp_path / "note.md").write_text("# Note\n\n[[definitely-not-a-real-file]]\n")
    md_files = dreaming.list_md_files(tmp_path)
    result = dreaming.pass_broken_wikilinks(tmp_path, md_files)
    assert {"file": "note.md", "link": "definitely-not-a-real-file"} in result["broken"]


# --- Denoise: structural same-title suppression ------------------------------

def test_near_duplicates_suppresses_bo_spec_build_log_pair(dreaming, tmp_path):
    specs = tmp_path / "Personal" / "Build Orchestrator" / "specs"
    logs = tmp_path / "Personal" / "Build Orchestrator" / "build-logs"
    specs.mkdir(parents=True)
    logs.mkdir(parents=True)
    (specs / "auto-2026-07-07-HSI-stop-too-tight-scores.md").write_text("# Same Build Title\n")
    (logs / "auto-20260707-stop-too-tight-scores-output.md").write_text("# Same Build Title\n")

    md_files = dreaming.list_md_files(tmp_path)
    result = dreaming.pass_near_duplicates(tmp_path, md_files)
    assert result["title_matches"] == []


def test_near_duplicates_keeps_non_bo_same_title_pair(dreaming, vault):
    md_files = dreaming.list_md_files(vault)
    result = dreaming.pass_near_duplicates(vault, md_files)
    assert any(set(tm["files"]) == {"dup-a.md", "dup-b.md"} for tm in result["title_matches"])


# --- Autofix -------------------------------------------------------------

@pytest.fixture
def autofix_vault(tmp_path):
    """A file with a punctuation-drifted link to 'Real File.md', plus an
    unfixable genuinely-dead link and an _Archive/ file that must stay untouched."""
    (tmp_path / "Real File.md").write_text("---\nupdated: 2026-01-01\n---\n\n# Real File\n")
    (tmp_path / "citing-note.md").write_text(
        "---\nupdated: 2026-01-01\n---\n\n# Citing Note\n\nSee [[real-file]] for details.\n"
    )
    (tmp_path / "dead-link-note.md").write_text("# Dead Link Note\n\n[[nonexistent-thing]]\n")
    archive = tmp_path / "_Archive"
    archive.mkdir()
    (archive / "old.md").write_text("# Old\n\n[[real-file]]\n")
    return tmp_path


def test_find_autofix_candidates_matches_punctuation_drift_only(dreaming, autofix_vault):
    md_files = dreaming.list_md_files(autofix_vault)
    candidates = dreaming.find_autofix_candidates(autofix_vault, md_files)
    assert len(candidates) == 1
    c = candidates[0]
    assert c["file"] == "citing-note.md"
    assert c["old_target"] == "real-file"
    assert c["new_target"] == "Real File"


def test_apply_autofix_backs_up_fixes_and_bumps_updated(dreaming, autofix_vault):
    md_files = dreaming.list_md_files(autofix_vault)
    candidates = dreaming.find_autofix_candidates(autofix_vault, md_files)
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)

    fixed_lines, backup_dir = dreaming.apply_autofix(autofix_vault, candidates, now)

    assert len(fixed_lines) == 1
    assert "citing-note.md:7" in fixed_lines[0]
    assert backup_dir.exists()
    assert (backup_dir / "citing-note.md").read_text() == (
        "---\nupdated: 2026-01-01\n---\n\n# Citing Note\n\nSee [[real-file]] for details.\n"
    )

    fixed_content = (autofix_vault / "citing-note.md").read_text()
    assert "[[Real File]]" in fixed_content
    assert "updated: 2026-08-01\n---\n" in fixed_content  # newline before closing fence must survive


def test_apply_autofix_is_idempotent(dreaming, autofix_vault):
    md_files = dreaming.list_md_files(autofix_vault)
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    dreaming.apply_autofix(autofix_vault, dreaming.find_autofix_candidates(autofix_vault, md_files), now)

    md_files_2 = dreaming.list_md_files(autofix_vault)
    second_candidates = dreaming.find_autofix_candidates(autofix_vault, md_files_2)
    fixed_lines_2, _ = dreaming.apply_autofix(autofix_vault, second_candidates, now)
    assert second_candidates == []
    assert fixed_lines_2 == []


def test_find_autofix_candidates_ignores_archive_files(dreaming, autofix_vault):
    md_files = dreaming.list_md_files(autofix_vault)
    candidates = dreaming.find_autofix_candidates(autofix_vault, md_files)
    assert not any(c["file"].startswith("_Archive/") for c in candidates)


def test_find_autofix_candidates_leaves_genuinely_dead_link(dreaming, autofix_vault):
    md_files = dreaming.list_md_files(autofix_vault)
    candidates = dreaming.find_autofix_candidates(autofix_vault, md_files)
    assert not any(c["old_target"] == "nonexistent-thing" for c in candidates)
