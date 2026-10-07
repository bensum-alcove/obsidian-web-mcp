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


# --- _keyword_leg tokenization tests -------------------------------------------

_LONG_SENTENCE = "Does a BS_BRAIN_API_KEY environment variable exist anywhere in the infrastructure?"


def test_keyword_leg_short_query_passes_raw_query_unchanged(vault_dir, monkeypatch):
    """<=4 content tokens: identical ripgrep call to pre-tokenization behaviour."""
    calls = []

    def fake_ripgrep(query, search_path, file_pattern, max_results, context_lines):
        calls.append(query)
        return []

    monkeypatch.setattr(query_tool.shutil, "which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr(query_tool, "_search_ripgrep", fake_ripgrep)

    query_tool._keyword_leg("supervisord config path", "*.md", 50)

    assert calls == ["supervisord config path"]


def test_keyword_leg_fallback_triggers_on_weak_nonempty_result(vault_dir, monkeypatch):
    """A non-empty but too-small primary result (< _WEAK_MATCH_MIN_COUNT) must still
    trigger the per-token merge -- not only a zero-match result."""
    (vault_dir / "correct.md").write_text(
        "The BS_BRAIN_API_KEY environment variable lives in supervisord config.\n"
    )
    (vault_dir / "adjacent.md").write_text(
        "This document discusses infrastructure history broadly.\n"
    )

    def fake_ripgrep(query, search_path, file_pattern, max_results, context_lines):
        # Simulates an incidental regex hit on an unrelated file: one match, low signal.
        return [{
            "path": "adjacent.md",
            "line_number": 1,
            "match_context": "This document discusses infrastructure history broadly.",
        }]

    monkeypatch.setattr(query_tool.shutil, "which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr(query_tool, "_search_ripgrep", fake_ripgrep)

    result = query_tool._keyword_leg(_LONG_SENTENCE, "*.md", 50)
    paths = [p for p, _ in result]
    assert "correct.md" in paths


def test_keyword_leg_fallback_triggers_when_no_match_has_multi_token_overlap(vault_dir, monkeypatch):
    """>= _WEAK_MATCH_MIN_COUNT matches, but none overlaps more than one query token,
    must still be treated as weak and trigger the per-token merge."""
    (vault_dir / "correct.md").write_text(
        "The BS_BRAIN_API_KEY environment variable lives in supervisord config.\n"
    )
    (vault_dir / "n1.md").write_text("infrastructure notes one\n")
    (vault_dir / "n2.md").write_text("infrastructure notes two\n")
    (vault_dir / "n3.md").write_text("infrastructure notes three\n")

    def fake_ripgrep(query, search_path, file_pattern, max_results, context_lines):
        return [
            {"path": "n1.md", "line_number": 1, "match_context": "infrastructure notes one"},
            {"path": "n2.md", "line_number": 1, "match_context": "infrastructure notes two"},
            {"path": "n3.md", "line_number": 1, "match_context": "infrastructure notes three"},
        ]

    monkeypatch.setattr(query_tool.shutil, "which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr(query_tool, "_search_ripgrep", fake_ripgrep)

    result = query_tool._keyword_leg(_LONG_SENTENCE, "*.md", 50)
    paths = [p for p, _ in result]
    assert "correct.md" in paths


def test_keyword_leg_keeps_strong_primary_result(vault_dir, monkeypatch):
    """A primary result with enough matches and real multi-token overlap is trusted
    as-is -- the per-token merge must not override a genuinely strong result."""
    fixed_matches = [
        {"path": "a.md", "line_number": 1, "match_context": "BS_BRAIN_API_KEY environment variable"},
        {"path": "b.md", "line_number": 1, "match_context": "environment variable infrastructure"},
        {"path": "c.md", "line_number": 1, "match_context": "infrastructure environment exist"},
    ]

    def fake_ripgrep(query, search_path, file_pattern, max_results, context_lines):
        return fixed_matches

    monkeypatch.setattr(query_tool.shutil, "which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr(query_tool, "_search_ripgrep", fake_ripgrep)

    result = query_tool._keyword_leg(_LONG_SENTENCE, "*.md", 50)
    assert result == [("a.md", 1), ("b.md", 1), ("c.md", 1)]


def test_keyword_leg_kill_switch_off_ignores_token_count_and_weakness(vault_dir, monkeypatch):
    """Kill switch disabled: raw sentence passed straight through, unchanged from
    pre-tokenization behaviour, even for a long query with an empty primary result."""
    monkeypatch.setattr(config, "VAULT_QUERY_KEYWORD_TOKENIZE", False)
    calls = []

    def fake_ripgrep(query, search_path, file_pattern, max_results, context_lines):
        calls.append(query)
        return []

    monkeypatch.setattr(query_tool.shutil, "which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr(query_tool, "_search_ripgrep", fake_ripgrep)

    query_tool._keyword_leg(_LONG_SENTENCE, "*.md", 50)

    assert calls == [_LONG_SENTENCE]


def test_tokenize_query_and_search_by_tokens_importable_from_query_module():
    """query.py imports these directly from search.py -- guards against a stale import."""
    assert query_tool._tokenize_query is not None
    assert query_tool._search_by_tokens is not None


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


def _build_bo_fixture(vault_dir):
    bo = vault_dir / "Personal" / "Build Orchestrator" / "specs"
    bo.mkdir(parents=True)
    (bo / "bo-harness-final-v2.md").write_text(
        "bo-harness-final-v2 cc-guard-mod-v2 bo-goal-hygiene-v1 current state and next action\n"
        "bo_action verbs list retry_review review_transport_exhausted operator recovery\n"
    )
    (vault_dir / "Personal" / "hot.md").write_text(
        "# Personal hot\nBuild Orchestrator live state lives in Infrastructure/hot.md.\n"
    )
    infra = vault_dir / "BS 2nd Brain" / "Alcove" / "Infrastructure"
    infra.mkdir(parents=True)
    (infra / "hot.md").write_text(
        "---\ncovers:\n- Personal/Build Orchestrator/\n- BS 2nd Brain/Alcove/Infrastructure/\n---\n# Infra hot\n"
    )


@pytest.mark.parametrize("question", [
    "bo-harness-final-v2 cc-guard-mod-v2 bo-goal-hygiene-v1 current state and next action",
    "bo_action verbs list retry_review review_transport_exhausted operator recovery",
])
def test_vault_answer_context_covers_outranks_shared_folder(vault_dir, question):
    _build_bo_fixture(vault_dir)
    result = json.loads(query_tool.vault_answer_context(question))
    hot = result["hot"]
    assert hot[0]["path"] == "BS 2nd Brain/Alcove/Infrastructure/hot.md"
    assert hot[0]["selected_by"] == "covers"
    by_path = {h["path"]: h["selected_by"] for h in hot}
    assert by_path["Personal/hot.md"] == "shared_folder"


def test_vault_answer_context_without_covers_uses_shared_folder(vault_dir):
    skills_dir = vault_dir / "Skills"
    skills_dir.mkdir()
    (skills_dir / "hot.md").write_text("Hot context for Skills.\n")
    (skills_dir / "answer-marker.md").write_text("answer-context-unique-marker content\n")
    other = vault_dir / "Other"
    other.mkdir()
    (other / "hot.md").write_text("Other hot.\n")

    result = json.loads(query_tool.vault_answer_context("answer-context-unique-marker"))
    assert result["hot"][0]["path"] == "Skills/hot.md"
    assert result["hot"][0]["selected_by"] == "shared_folder"
    assert {h["path"]: h["selected_by"] for h in result["hot"]}["Other/hot.md"] == "fallback"


REAL_PERSONAL_HOT = """---
type: hot-cache
format: compiled-truth
char_budget: 5000
timeline: Personal/hot-timeline/hot-timeline-through-2026-10-07.md
tags:
- hot-cache
- personal
- session-state
updated: '2026-10-07'
---

# Personal — Current State

> **Compiled truth only** (`BS 2nd Brain/_SCHEMA.md` Write Rule 21). Rewrite in place; no dated diary entries.

## Where current state actually lives
- **Build Orchestrator** (specs and logs are filed under `Personal/Build Orchestrator/`): live state is in `BS 2nd Brain/Alcove/Infrastructure/hot.md`. Don't use this page for BO state.
- **MahMah's Estate:** `BS 2nd Brain/MahMahs Estate/hot.md`.

## Personal matters in flight
- Nothing recorded yet.
"""


def _realistic_infra_hot(body_chars: int, char_budget: int | None = None) -> str:
    """Infrastructure hot.md shaped like the live one: covers: frontmatter, long compiled-truth body."""
    fm = ["---", "type: hot-cache", "format: compiled-truth"]
    if char_budget is not None:
        fm.append(f"char_budget: {char_budget}")
    fm += [
        "covers:",
        "- Personal/Build Orchestrator/",
        "- BS 2nd Brain/Alcove/Infrastructure/",
        "tags:",
        "- hot-cache",
        "- infrastructure",
        "---",
        "",
    ]
    body = "# Infrastructure — Current State\n\n## Build Orchestrator\n"
    line = "- Live state item: build chain, deploy gate and next operator action recorded here.\n"
    while len(body) < body_chars:
        body += line
    return "\n".join(fm) + body[:body_chars]


def _build_realistic_bo_fixture(vault_dir, infra_text):
    bo = vault_dir / "Personal" / "Build Orchestrator" / "specs"
    bo.mkdir(parents=True)
    (bo / "bo-harness-final-v2.md").write_text(
        "bo-harness-final-v2 cc-guard-mod-v2 bo-goal-hygiene-v1 current state and next action\n"
        "bo_action verbs list retry_review review_transport_exhausted operator recovery\n"
    )
    (vault_dir / "Personal" / "hot.md").write_text(REAL_PERSONAL_HOT)
    infra = vault_dir / "BS 2nd Brain" / "Alcove" / "Infrastructure"
    infra.mkdir(parents=True)
    (infra / "hot.md").write_text(infra_text)


WITNESS_QUESTIONS = [
    "bo-harness-final-v2 cc-guard-mod-v2 bo-goal-hygiene-v1 current state and next action",
    "bo_action verbs list retry_review review_transport_exhausted operator recovery",
]


@pytest.mark.parametrize("question", WITNESS_QUESTIONS)
def test_vault_answer_context_realistic_5kb_infra_hot_md_selected_by_covers(vault_dir, question):
    # ~5.3KB on disk (over the old 3,072-byte cap) but under the 5,000-char Rule 21 budget.
    text = _realistic_infra_hot(body_chars=4850)
    assert 5000 < len(text.encode()) < 5600
    _build_realistic_bo_fixture(vault_dir, text)

    result = json.loads(query_tool.vault_answer_context(question))
    hot = result["hot"]
    assert hot[0]["path"] == "BS 2nd Brain/Alcove/Infrastructure/hot.md"
    assert hot[0]["selected_by"] == "covers"
    assert hot[0]["content"] == text
    assert "truncated" not in hot[0]
    assert not [w for w in result["warnings"] if w["reason"] == "over_budget"]


@pytest.mark.parametrize("question", WITNESS_QUESTIONS)
def test_vault_answer_context_over_budget_covers_hot_md_is_truncated_not_dropped(vault_dir, question):
    text = _realistic_infra_hot(body_chars=7000)
    _build_realistic_bo_fixture(vault_dir, text)

    result = json.loads(query_tool.vault_answer_context(question))
    hot = result["hot"]
    assert hot[0]["path"] == "BS 2nd Brain/Alcove/Infrastructure/hot.md"
    assert hot[0]["selected_by"] == "covers"
    assert hot[0]["truncated"] is True
    assert hot[0]["content"].startswith("---\n")
    assert len(hot[0]["content"]) < len(text)
    assert text.startswith(hot[0]["content"])
    warnings = [w for w in result["warnings"] if w["reason"] == "over_budget"]
    assert [w["path"] for w in warnings] == ["BS 2nd Brain/Alcove/Infrastructure/hot.md"]


def test_vault_answer_context_honours_frontmatter_char_budget(vault_dir):
    # 4,000-char body is under the default budget but over this file's own budget.
    _build_realistic_bo_fixture(vault_dir, _realistic_infra_hot(body_chars=4000, char_budget=1000))
    result = json.loads(query_tool.vault_answer_context(WITNESS_QUESTIONS[0]))
    assert result["hot"][0]["truncated"] is True
    assert any(w["reason"] == "over_budget" for w in result["warnings"])


def test_vault_answer_context_over_budget_non_covers_hot_md_is_skipped_with_warning(vault_dir):
    skills_dir = vault_dir / "Skills"
    skills_dir.mkdir()
    (skills_dir / "hot.md").write_text("x" * 6000)
    (skills_dir / "answer-marker.md").write_text("answer-context-unique-marker content\n")

    result = json.loads(query_tool.vault_answer_context("answer-context-unique-marker"))
    assert "Skills/hot.md" not in [h["path"] for h in result["hot"]]
    skipped = [w for w in result["warnings"] if w["reason"] == "hot_skipped_over_budget"]
    assert [w["path"] for w in skipped] == ["Skills/hot.md"]


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


# --- vault-query-calibration-v2: RRF_K / canonical-boost tests ----------------

def test_rrf_fuse_default_k_reads_from_config(monkeypatch):
    """No explicit k -> config.VAULT_QUERY_RRF_K, not the old hardcoded 60."""
    monkeypatch.setattr(config, "VAULT_QUERY_RRF_K", 5.0)
    scores = query_tool._rrf_fuse(["a.md"], [])
    assert scores["a.md"] == pytest.approx(1 / 6)


def test_rrf_fuse_kill_switch_at_k60_matches_pre_calibration_behaviour(monkeypatch):
    """Kill switch: explicitly setting k back to the old hardcoded RRF_K=60 is
    byte-identical to pre-calibration behaviour."""
    monkeypatch.setattr(config, "VAULT_QUERY_RRF_K", 60.0)
    scores = query_tool._rrf_fuse(["a.md"], [], k=None)
    assert scores["a.md"] == pytest.approx(1 / 61)


def test_rrf_fuse_default_matches_r5_085_v1_calibration():
    """vault-retrieval-r5-085-v1: default k lowered from 60 to 6 -- empirically
    swept against frozen-v3 (aggregate, not per-question), stable plateau at
    k=4..8. Pins the new default so a future change is deliberate."""
    assert config.VAULT_QUERY_RRF_K == 6.0
    scores = query_tool._rrf_fuse(["a.md"], [], k=None)
    assert scores["a.md"] == pytest.approx(1 / 7)


def test_canonical_boost_factor_kill_switch_off_is_noop(vault_dir, monkeypatch):
    """Explicitly setting VAULT_QUERY_CANONICAL_BOOST=1.0 -> no-op regardless of
    frontmatter (the kill switch), independent of whatever the live default is."""
    canonical_dir = vault_dir / "Canonical State" / "records"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "widget.md").write_text("---\ntype: canonical-state\n---\n\nWidget.\n")

    monkeypatch.setattr(config, "VAULT_QUERY_CANONICAL_BOOST", 1.0)
    factor = query_tool._canonical_boost_factor("Canonical State/records/widget.md")
    assert factor == 1.0


def test_canonical_boost_factor_default_matches_r5_085_v1_calibration(vault_dir):
    """vault-retrieval-r5-085-v1: default boost raised from 1.0 (and the
    previously live-only 1.3, which was never promoted to this default and was
    empirically almost a no-op) to 3.0 -- empirically swept against frozen-v3,
    plateau at 3.0-5.0. Pins the new default so a future change is deliberate."""
    canonical_dir = vault_dir / "Canonical State" / "records"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "widget.md").write_text("---\ntype: canonical-state\n---\n\nWidget.\n")

    assert config.VAULT_QUERY_CANONICAL_BOOST == 3.0
    factor = query_tool._canonical_boost_factor("Canonical State/records/widget.md")
    assert factor == 3.0


def test_canonical_boost_factor_boosts_canonical_state_frontmatter(vault_dir, monkeypatch):
    monkeypatch.setattr(config, "VAULT_QUERY_CANONICAL_BOOST", 1.3)
    canonical_dir = vault_dir / "Canonical State" / "records"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "widget.md").write_text("---\ntype: canonical-state\n---\n\nWidget.\n")

    factor = query_tool._canonical_boost_factor("Canonical State/records/widget.md")
    assert factor == 1.3


def test_canonical_boost_factor_ignores_non_canonical_frontmatter(vault_dir, monkeypatch):
    monkeypatch.setattr(config, "VAULT_QUERY_CANONICAL_BOOST", 1.3)
    canonical_dir = vault_dir / "Canonical State" / "records"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "not-canonical.md").write_text("---\ntype: reference\n---\n\nJust a reference doc.\n")

    factor = query_tool._canonical_boost_factor("Canonical State/records/not-canonical.md")
    assert factor == 1.0


def test_canonical_boost_factor_skips_frontmatter_parse_outside_canonical_dir(vault_dir, monkeypatch):
    """Path-prefilter: a file outside Canonical State/records/ never reaches the
    frontmatter parser, even if it happens to declare type: canonical-state --
    this is the latency shortcut, not a correctness path."""
    monkeypatch.setattr(config, "VAULT_QUERY_CANONICAL_BOOST", 1.3)
    calls = []
    monkeypatch.setattr(
        query_tool, "_frontmatter_type", lambda p: (calls.append(p), "canonical-state")[1]
    )
    (vault_dir / "elsewhere.md").write_text("---\ntype: canonical-state\n---\n\nElsewhere.\n")

    factor = query_tool._canonical_boost_factor("elsewhere.md")
    assert factor == 1.0
    assert calls == []


def test_vault_query_canonical_boost_promotes_canonical_record(vault_dir, monkeypatch):
    """Integration: with the boost enabled, a canonical-state record that ties on
    keyword rank with a non-canonical file scores higher and sorts first."""
    monkeypatch.setattr(config, "VAULT_QUERY_CANONICAL_BOOST", 1.3)
    canonical_dir = vault_dir / "Canonical State" / "records"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "widget-state.md").write_text(
        "---\ntype: canonical-state\n---\n\nboost-tie-marker widget status.\n"
    )
    (vault_dir / "widget-prose.md").write_text("boost-tie-marker widget status prose.\n")

    result = json.loads(query_tool.vault_query("boost-tie-marker widget status"))
    scores = {r["path"]: r["score"] for r in result["results"]}
    assert scores["Canonical State/records/widget-state.md"] > scores["widget-prose.md"]


# --- vault-retrieval-candidate-recall-v1 unique tests (dev checkout) ---------
# Ported in during checkout reconciliation (vault-retrieval-r5-085-v1). These
# pin the paraphrase-recall fix's specific behaviour (allow_partial keyword
# leg, raised candidate-pool depth) that the live checkout's suite above does
# not cover.

def test_rrf_fuse_consensus_beats_single_leg_rank_one():
    scores = query_tool._rrf_fuse(["single_leg_winner.md", "x.md"], ["x.md", "y.md"])
    # x.md appears rank-2 keyword + rank-1 semantic; single_leg_winner.md is
    # rank-1 keyword only. This is the diagnosed "flat RRF" dynamic itself
    # (not something this build changes) -- test just pins current behaviour.
    assert "single_leg_winner.md" in scores
    assert "x.md" in scores


def test_keyword_leg_allow_partial_recovers_candidate_absent_under_strict_and(tmp_path, monkeypatch):
    """Direct regression test for the diagnosed paraphrase failure: a long,
    natural-language query whose tokens are individually scattered such that
    no single small file satisfies every token, but the correct answer
    document satisfies most of them, must still appear as a keyword-leg
    candidate once allow_partial is enabled."""
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "RETRIEVAL_EXCLUDED_DIRS", set())
    monkeypatch.setattr(config, "VAULT_QUERY_KEYWORD_TOKENIZE", True)
    monkeypatch.setattr(config, "VAULT_QUERY_ALLOW_PARTIAL_KEYWORD_MATCH", True)

    (tmp_path / "brain-dashboard.md").write_text(
        "Public dashboard at brain.bensum.org, backend on port 8432.\n"
    )
    (tmp_path / "infrastructure-changelog.md").write_text(
        " ".join(["filler"] * 200 + [
            "network", "port", "serves", "public", "dashboard",
            "people", "reach", "brain.bensum.org",
        ]) + "\n"
    )

    query = "Which network port serves the public dashboard people reach at brain.bensum.org?"
    hits = query_tool._keyword_leg(query, "*.md", 150)
    paths = [p for p, _ in hits]
    assert "brain-dashboard.md" in paths


def test_keyword_leg_allow_partial_disabled_reproduces_old_bug(tmp_path, monkeypatch):
    """Confirms the kill switch: with allow_partial disabled, the diagnosed
    bug reproduces exactly (candidate absent under the strict AND gate)."""
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "RETRIEVAL_EXCLUDED_DIRS", set())
    monkeypatch.setattr(config, "VAULT_QUERY_KEYWORD_TOKENIZE", True)
    monkeypatch.setattr(config, "VAULT_QUERY_ALLOW_PARTIAL_KEYWORD_MATCH", False)

    (tmp_path / "brain-dashboard.md").write_text(
        "Public dashboard at brain.bensum.org, backend on port 8432.\n"
    )
    (tmp_path / "infrastructure-changelog.md").write_text(
        " ".join(["filler"] * 200 + [
            "network", "port", "serves", "public", "dashboard",
            "people", "reach", "brain.bensum.org",
        ]) + "\n"
    )

    query = "Which network port serves the public dashboard people reach at brain.bensum.org?"
    hits = query_tool._keyword_leg(query, "*.md", 150)
    paths = [p for p, _ in hits]
    assert "brain-dashboard.md" not in paths


def test_vault_query_fetch_depth_uses_config_floor(monkeypatch):
    """vault-retrieval-candidate-recall-v1: fetch_n must respect the raised
    VAULT_SEMANTIC_FETCH_MIN floor, not just top_k*10, for small top_k callers."""
    monkeypatch.setattr(config, "VAULT_SEMANTIC_FETCH_MIN", 100)
    # top_k=5 -> top_k*10=50, but floor (as used inside vault_query) is max(100, 50)=100
    fetch_n = min(300, max(config.VAULT_SEMANTIC_FETCH_MIN, 5 * 10))
    assert fetch_n == 100
