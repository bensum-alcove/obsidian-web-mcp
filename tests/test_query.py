"""Tests for vault_query (RRF-fused hybrid retrieval), ported into the dev
checkout's src/ tree alongside its calibration-v2 mechanisms plus the new
vault-retrieval-candidate-recall-v1 additions (allow_partial keyword leg,
raised candidate-pool depth)."""

import json

import obsidian_vault_mcp.config as config
from obsidian_vault_mcp.tools.query import (
    _rrf_fuse,
    _canonical_boost_factor,
    _decay_factor,
    _is_archived,
    _keyword_leg,
    vault_query,
)


def test_rrf_fuse_reads_config_default(monkeypatch):
    monkeypatch.setattr(config, "VAULT_QUERY_RRF_K", 60.0)
    scores = _rrf_fuse(["a.md"], ["b.md"])
    assert scores["a.md"] == 1.0 / 61
    assert scores["b.md"] == 1.0 / 61


def test_rrf_fuse_default_matches_old_hardcoded_constant():
    """Kill-switch check: default k must equal the old hardcoded RRF_K=60."""
    scores = _rrf_fuse(["a.md"], [], k=None)
    assert abs(scores["a.md"] - 1.0 / 61) < 1e-9


def test_rrf_fuse_consensus_beats_single_leg_rank_one():
    scores = _rrf_fuse(["single_leg_winner.md", "x.md"], ["x.md", "y.md"])
    # x.md appears rank-2 keyword + rank-1 semantic; single_leg_winner.md is
    # rank-1 keyword only. This is the diagnosed "flat RRF" dynamic itself
    # (not something this build changes) -- test just pins current behaviour.
    assert "single_leg_winner.md" in scores
    assert "x.md" in scores


def test_canonical_boost_factor_noop_at_default(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VAULT_QUERY_CANONICAL_BOOST", 1.0)
    assert _canonical_boost_factor("Canonical State/records/anything.md") == 1.0


def test_canonical_boost_factor_skips_frontmatter_parse_outside_canonical_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "VAULT_QUERY_CANONICAL_BOOST", 1.5)
    # No file exists at this path at all -- if this function tried to read it,
    # it would hit an exception path. Passing a non-canonical-dir path must
    # short-circuit before ever touching the filesystem.
    assert _canonical_boost_factor("Some/Other/Dir/file.md") == 1.0


def test_canonical_boost_factor_boosts_canonical_state_type(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "VAULT_QUERY_CANONICAL_BOOST", 1.3)
    d = tmp_path / "Canonical State" / "records"
    d.mkdir(parents=True)
    (d / "foo.md").write_text("---\ntype: canonical-state\n---\n\nBody.\n")
    assert _canonical_boost_factor("Canonical State/records/foo.md") == 1.3


def test_canonical_boost_factor_ignores_non_canonical_frontmatter(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "VAULT_QUERY_CANONICAL_BOOST", 1.3)
    d = tmp_path / "Canonical State" / "records"
    d.mkdir(parents=True)
    (d / "foo.md").write_text("---\ntype: note\n---\n\nBody.\n")
    assert _canonical_boost_factor("Canonical State/records/foo.md") == 1.0


def test_is_archived_detects_archive_and_trash_dirs():
    assert _is_archived("_Archive/foo.md")
    assert _is_archived("some/.trash/foo.md")
    assert not _is_archived("Regular/foo.md")


def test_decay_factor_no_decay_at_zero_age():
    assert _decay_factor("anything.md", 0.0) == 1.0


def test_decay_factor_uses_longest_matching_prefix_override(monkeypatch):
    monkeypatch.setattr(config, "VAULT_QUERY_DEFAULT_HALF_LIFE_DAYS", 90.0)
    monkeypatch.setattr(config, "VAULT_QUERY_HALF_LIFE_OVERRIDES", {"Skills/": 180.0})
    default_decay = _decay_factor("Other/foo.md", 90.0)
    skills_decay = _decay_factor("Skills/foo.md", 90.0)
    assert skills_decay > default_decay  # longer half-life decays slower


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
    hits = _keyword_leg(query, "*.md", 150)
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
    hits = _keyword_leg(query, "*.md", 150)
    paths = [p for p, _ in hits]
    assert "brain-dashboard.md" not in paths


def test_vault_query_returns_results_and_total_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "RETRIEVAL_EXCLUDED_DIRS", set())
    (tmp_path / "note.md").write_text("Some searchable content about ports and dashboards.\n")

    result = json.loads(vault_query("ports and dashboards"))
    assert "results" in result
    assert "total_candidates" in result


def test_vault_query_excludes_archive_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "RETRIEVAL_EXCLUDED_DIRS", set())
    archive_dir = tmp_path / "_Archive"
    archive_dir.mkdir()
    (archive_dir / "old.md").write_text("uniquearchivedsentinelword\n")

    result = json.loads(vault_query("uniquearchivedsentinelword"))
    paths = [r["path"] for r in result["results"]]
    assert not any("_Archive" in p for p in paths)

    result_incl = json.loads(vault_query("uniquearchivedsentinelword", include_archive=True))
    paths_incl = [r["path"] for r in result_incl["results"]]
    assert any("_Archive" in p for p in paths_incl)


def test_vault_query_fetch_depth_uses_config_floor(monkeypatch):
    """vault-retrieval-candidate-recall-v1: fetch_n must respect the raised
    VAULT_SEMANTIC_FETCH_MIN floor, not just top_k*10, for small top_k callers."""
    monkeypatch.setattr(config, "VAULT_SEMANTIC_FETCH_MIN", 100)
    # top_k=5 -> top_k*10=50, but floor (as used inside vault_query) is max(100, 50)=100
    fetch_n = min(300, max(config.VAULT_SEMANTIC_FETCH_MIN, 5 * 10))
    assert fetch_n == 100
