"""Tests for vault_search's tokenized-augmentation fallback (tools/search.py:
_augment_with_tokenized_matches, gated by config.VAULT_SEARCH_TOKENIZE). See
tools/query.py's _keyword_leg for the equivalent, separately-switched behaviour
on vault_query, and test_search_tokenize.py for _tokenize_query/_search_by_tokens
unit coverage this build reuses rather than duplicating."""

import json

from obsidian_vault_mcp import config
from obsidian_vault_mcp.tools import search as search_tool
from obsidian_vault_mcp.tools.search import vault_search

_LONG_QUERY = "Does a BS_BRAIN_API_KEY environment variable exist anywhere in the infrastructure?"


def test_short_query_identical_call_and_output(vault_dir, monkeypatch):
    """<=4 content tokens: identical ripgrep call and output to pre-change behaviour."""
    calls = []

    def fake_ripgrep(query, search_path, file_pattern, max_results, context_lines):
        calls.append(query)
        return []

    monkeypatch.setattr(search_tool.shutil, "which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr(search_tool, "_search_ripgrep", fake_ripgrep)

    result = json.loads(vault_search("supervisord config path"))

    assert calls == ["supervisord config path"]
    assert result["results"] == []
    assert result["total_matches"] == 0


def test_short_query_zero_literal_still_uses_old_fallback_no_match_type(vault_dir, monkeypatch):
    """<=4 tokens, zero literal matches: old simple-split fallback still runs
    untouched, and no match_type field appears anywhere (byte-identical output
    shape to pre-change)."""
    def fake_ripgrep(query, search_path, file_pattern, max_results, context_lines):
        return []

    monkeypatch.setattr(search_tool.shutil, "which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr(search_tool, "_search_ripgrep", fake_ripgrep)

    (vault_dir / "alpha.md").write_text("alpha beta content\n")

    result = json.loads(vault_search("alpha beta"))
    assert result["total_matches"] >= 1
    assert all("match_type" not in m for m in result["results"])


def test_long_query_literal_matches_kept_ahead_of_tokenized(vault_dir, monkeypatch):
    """Literal matches are never reordered, demoted, or dropped -- they stay
    first, in their original order, tagged 'literal'; tokenized matches for
    other files are appended below, tagged 'tokenized'."""
    def fake_ripgrep(query, search_path, file_pattern, max_results, context_lines):
        return [{"path": "literal-hit.md", "line_number": 1, "match_context": query}]

    monkeypatch.setattr(search_tool.shutil, "which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr(search_tool, "_search_ripgrep", fake_ripgrep)

    (vault_dir / "literal-hit.md").write_text("placeholder\n")
    (vault_dir / "token-hit.md").write_text(
        "BS_BRAIN_API_KEY environment variable infrastructure exists here\n"
    )

    result = json.loads(vault_search(_LONG_QUERY))
    paths = [m["path"] for m in result["results"]]

    assert paths[0] == "literal-hit.md"
    assert result["results"][0]["match_type"] == "literal"
    assert "token-hit.md" in paths
    token_entries = [m for m in result["results"] if m["path"] == "token-hit.md"]
    assert all(m["match_type"] == "tokenized" for m in token_entries)


def test_long_query_zero_literal_matches_all_tokenized(vault_dir, monkeypatch):
    """Zero literal matches for a long query: tokenized results are returned,
    every one marked 'tokenized' (no unmarked/legacy entries mixed in)."""
    def fake_ripgrep(query, search_path, file_pattern, max_results, context_lines):
        return []

    monkeypatch.setattr(search_tool.shutil, "which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr(search_tool, "_search_ripgrep", fake_ripgrep)

    (vault_dir / "token-hit.md").write_text(
        "BS_BRAIN_API_KEY environment variable infrastructure exists here\n"
    )

    result = json.loads(vault_search(_LONG_QUERY))
    assert result["total_matches"] >= 1
    assert all(m["match_type"] == "tokenized" for m in result["results"])


def test_exact_string_contract_literal_present_and_absent(vault_dir, monkeypatch):
    """Exact-string verification contract: a distinctive literal match exists in
    exactly one file -> that file comes back match_type == 'literal'. When the
    literal search finds nothing, match_type == 'literal' entries are zero even
    though tokenized results (from the query's other tokens) still appear."""
    def fake_ripgrep_hit(query, search_path, file_pattern, max_results, context_lines):
        return [{"path": "exact.md", "line_number": 1, "match_context": query}]

    def fake_ripgrep_miss(query, search_path, file_pattern, max_results, context_lines):
        return []

    monkeypatch.setattr(search_tool.shutil, "which", lambda name: "/usr/bin/rg")

    monkeypatch.setattr(search_tool, "_search_ripgrep", fake_ripgrep_hit)
    present = json.loads(vault_search(_LONG_QUERY))
    literal_present = [m for m in present["results"] if m["match_type"] == "literal"]
    assert literal_present and literal_present[0]["path"] == "exact.md"

    (vault_dir / "token-only.md").write_text(
        "BS_BRAIN_API_KEY environment variable infrastructure notes\n"
    )
    monkeypatch.setattr(search_tool, "_search_ripgrep", fake_ripgrep_miss)
    absent = json.loads(vault_search(_LONG_QUERY))
    literal_absent = [m for m in absent["results"] if m.get("match_type") == "literal"]
    assert literal_absent == []
    assert any(m["match_type"] == "tokenized" for m in absent["results"])


def test_atomic_identifier_survives_tokenized_augmentation(vault_dir, monkeypatch):
    """_tokenize_query is reused unchanged -- an underscored identifier in a long
    query still matches as one atomic token, same delimiter classes as the
    vault_query keyword-leg build."""
    def fake_ripgrep(query, search_path, file_pattern, max_results, context_lines):
        return []

    monkeypatch.setattr(search_tool.shutil, "which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr(search_tool, "_search_ripgrep", fake_ripgrep)

    (vault_dir / "identifier.md").write_text("BS_BRAIN_API_KEY lives here\n")

    result = json.loads(vault_search(_LONG_QUERY))
    paths = [m["path"] for m in result["results"]]
    assert "identifier.md" in paths


def test_kill_switch_off_byte_identical_output(vault_dir, monkeypatch):
    """VAULT_SEARCH_TOKENIZE=0: no augmentation, no match_type field, regardless
    of query length -- byte-identical to pre-change behaviour."""
    monkeypatch.setattr(config, "VAULT_SEARCH_TOKENIZE", False)

    def fake_ripgrep(query, search_path, file_pattern, max_results, context_lines):
        return []

    monkeypatch.setattr(search_tool.shutil, "which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr(search_tool, "_search_ripgrep", fake_ripgrep)

    (vault_dir / "token-hit.md").write_text(
        "BS_BRAIN_API_KEY environment variable infrastructure exists here\n"
    )

    result = json.loads(vault_search(_LONG_QUERY))
    assert all("match_type" not in m for m in result["results"])


def test_no_duplicate_paths_when_literal_and_tokenized_both_match(vault_dir, monkeypatch):
    """A file that matches both the literal search and the tokenized search
    appears exactly once, kept as its literal (higher-priority) entry."""
    def fake_ripgrep(query, search_path, file_pattern, max_results, context_lines):
        return [{"path": "both.md", "line_number": 1, "match_context": query}]

    monkeypatch.setattr(search_tool.shutil, "which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr(search_tool, "_search_ripgrep", fake_ripgrep)

    (vault_dir / "both.md").write_text(
        "BS_BRAIN_API_KEY environment variable infrastructure exists here\n"
    )

    result = json.loads(vault_search(_LONG_QUERY))
    paths = [m["path"] for m in result["results"]]
    assert paths.count("both.md") == 1
    assert result["results"][0]["match_type"] == "literal"
