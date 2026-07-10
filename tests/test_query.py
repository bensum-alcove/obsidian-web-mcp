"""Tests for tools/query.py -- vault_query (RRF hybrid) and vault_answer_context."""

import json
import os
import time

import pytest

from obsidian_vault_mcp import config
from obsidian_vault_mcp.tools import query as query_tool


# --- Pure function unit tests -------------------------------------------------

def test_rrf_fuse_combines_scores_from_both_legs():
    keyword = ["a.md", "b.md", "c.md"]
    semantic = ["b.md", "d.md"]
    scores = query_tool._rrf_fuse(keyword, semantic, k=60)

    assert scores["a.md"] == pytest.approx(1 / 61)
    assert scores["c.md"] == pytest.approx(1 / 63)
    assert scores["d.md"] == pytest.approx(1 / 62)
    # b.md appears in both legs (rank 2 keyword, rank 1 semantic) -> summed
    assert scores["b.md"] == pytest.approx(1 / 62 + 1 / 61)
    # present in both legs should outscore single-leg entries
    assert scores["b.md"] > scores["a.md"]


def test_rrf_fuse_empty_legs_returns_empty():
    assert query_tool._rrf_fuse([], []) == {}


def test_decay_factor_recent_file_near_one():
    factor = query_tool._decay_factor("Alcove/notes.md", age_days=0)
    assert factor == pytest.approx(1.0)


def test_decay_factor_decreases_with_age():
    recent = query_tool._decay_factor("Alcove/notes.md", age_days=1)
    old = query_tool._decay_factor("Alcove/notes.md", age_days=200)
    assert old < recent


def test_decay_factor_default_half_life_matches_config():
    hl = config.VAULT_QUERY_DEFAULT_HALF_LIFE_DAYS
    factor = query_tool._decay_factor("Alcove/notes.md", age_days=hl)
    assert factor == pytest.approx(0.5, rel=1e-6)


def test_decay_factor_prefix_override_decays_faster():
    hl_default = config.VAULT_QUERY_DEFAULT_HALF_LIFE_DAYS
    hl_prompts = config.VAULT_QUERY_HALF_LIFE_OVERRIDES["Claude-Code-Prompts/"]
    assert hl_prompts < hl_default

    age = 45
    default_factor = query_tool._decay_factor("Alcove/notes.md", age_days=age)
    prompt_factor = query_tool._decay_factor(
        "Alcove/Infrastructure/Claude-Code-Prompts/foo.md", age_days=age
    )
    assert prompt_factor < default_factor


def test_decay_factor_longest_prefix_wins():
    # "Clients/" (365d) should win over an unrelated default even though both could match
    factor_clients = query_tool._decay_factor("Alcove/Clients/foo.md", age_days=365)
    assert factor_clients == pytest.approx(0.5, rel=1e-6)


def test_is_archived_true_for_archive_and_trash():
    assert query_tool._is_archived("Alcove/_Archive/old-note.md") is True
    assert query_tool._is_archived(".trash/deleted.md") is True


def test_is_archived_false_for_normal_path():
    assert query_tool._is_archived("Alcove/Infrastructure/note.md") is False


# --- vault_query integration tests --------------------------------------------

def _touch_with_age(path, days_old: float):
    ts = time.time() - days_old * 86400
    os.utime(path, (ts, ts))


def test_vault_query_finds_keyword_match(vault_dir):
    result = json.loads(query_tool.vault_query("test note"))
    assert "error" not in result
    paths = [r["path"] for r in result["results"]]
    assert "test-note.md" in paths


def test_vault_query_stale_flag_true_for_old_file(vault_dir):
    target = vault_dir / "test-note.md"
    _touch_with_age(target, query_tool.STALE_DAYS + 10)

    result = json.loads(query_tool.vault_query("test note"))
    entry = next(r for r in result["results"] if r["path"] == "test-note.md")
    assert entry["stale"] is True


def test_vault_query_stale_flag_false_for_recent_file(vault_dir):
    result = json.loads(query_tool.vault_query("test note"))
    entry = next(r for r in result["results"] if r["path"] == "test-note.md")
    assert entry["stale"] is False


def test_vault_query_excludes_archive_by_default(vault_dir):
    archive_dir = vault_dir / "_Archive"
    archive_dir.mkdir()
    (archive_dir / "old.md").write_text("archived unique-marker-xyz content\n")

    result = json.loads(query_tool.vault_query("unique-marker-xyz"))
    paths = [r["path"] for r in result["results"]]
    assert not any("_Archive" in p for p in paths)


def test_vault_query_includes_archive_when_flagged(vault_dir):
    archive_dir = vault_dir / "_Archive"
    archive_dir.mkdir()
    (archive_dir / "old.md").write_text("archived unique-marker-xyz content\n")

    result = json.loads(query_tool.vault_query("unique-marker-xyz", include_archive=True))
    paths = [r["path"] for r in result["results"]]
    assert any("_Archive" in p for p in paths)


def test_vault_query_path_prefix_filters_results(vault_dir):
    result = json.loads(query_tool.vault_query("nested note", path_prefix="subfolder"))
    for r in result["results"]:
        assert r["path"].startswith("subfolder")


def test_vault_query_expand_handle_present_when_heading_found(vault_dir):
    (vault_dir / "headed.md").write_text(
        "---\ntype: note\n---\n\n## Marker Section\n\nunique-heading-marker text here.\n"
    )
    result = json.loads(query_tool.vault_query("unique-heading-marker"))
    entry = next(r for r in result["results"] if r["path"] == "headed.md")
    assert entry["heading"] == "Marker Section"
    assert entry["expand"] == {"path": "headed.md", "heading": "Marker Section"}


def test_vault_query_decay_true_by_default_lowers_old_file_score(vault_dir):
    (vault_dir / "old-match.md").write_text("decay-test-marker content\n")
    (vault_dir / "new-match.md").write_text("decay-test-marker content\n")
    _touch_with_age(vault_dir / "old-match.md", 200)

    with_decay = json.loads(query_tool.vault_query("decay-test-marker", decay=True))
    without_decay = json.loads(query_tool.vault_query("decay-test-marker", decay=False))

    old_score_decay = next(r["score"] for r in with_decay["results"] if r["path"] == "old-match.md")
    old_score_nodecay = next(r["score"] for r in without_decay["results"] if r["path"] == "old-match.md")
    assert old_score_decay < old_score_nodecay


# --- vault_answer_context integration tests -----------------------------------

def test_vault_answer_context_bundles_matching_hot_md(vault_dir):
    skills_dir = vault_dir / "Skills"
    skills_dir.mkdir()
    (skills_dir / "hot.md").write_text("Hot context for Skills.\n")
    (skills_dir / "answer-marker.md").write_text("answer-context-unique-marker content\n")

    result = json.loads(query_tool.vault_answer_context("answer-context-unique-marker"))
    hot_paths = [h["path"] for h in result["hot"]]
    assert "Skills/hot.md" in hot_paths


def test_vault_answer_context_skips_oversized_hot_md(vault_dir):
    skills_dir = vault_dir / "Skills"
    skills_dir.mkdir()
    (skills_dir / "hot.md").write_text("x" * (query_tool._HOT_MD_MAX_BYTES + 100))
    (skills_dir / "answer-marker.md").write_text("answer-context-unique-marker content\n")

    result = json.loads(query_tool.vault_answer_context("answer-context-unique-marker"))
    hot_paths = [h["path"] for h in result["hot"]]
    assert "Skills/hot.md" not in hot_paths


def test_vault_answer_context_warns_on_stale_result(vault_dir):
    target = vault_dir / "test-note.md"
    _touch_with_age(target, query_tool.STALE_DAYS + 10)

    result = json.loads(query_tool.vault_answer_context("test note"))
    stale_warnings = [w for w in result["warnings"] if w["path"] == "test-note.md" and w["reason"] == "stale"]
    assert len(stale_warnings) == 1


def test_vault_answer_context_warns_on_superseded_frontmatter(vault_dir):
    (vault_dir / "old-doc.md").write_text(
        "---\nstatus: superseded\n---\n\nsuperseded-content-marker text.\n"
    )
    result = json.loads(query_tool.vault_answer_context("superseded-content-marker"))
    superseded_warnings = [
        w for w in result["warnings"] if w["path"] == "old-doc.md" and w["reason"] == "superseded"
    ]
    assert len(superseded_warnings) == 1
