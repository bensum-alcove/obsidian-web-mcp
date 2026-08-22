"""Tests for vault_search/vault_query keyword-leg tokenization and the
allow_partial candidate-recall mechanism (vault-retrieval-candidate-recall-v1).
"""

import json

from obsidian_vault_mcp.tools.search import (
    _tokenize_query,
    _search_by_tokens,
    _search_keyword_fallback,
    _augment_with_tokenized_matches,
    vault_search,
)


def test_tokenize_query_drops_stopwords_and_interrogatives():
    tokens = _tokenize_query("What is the minimum score to trade on the system?")
    assert "what" not in [t.lower() for t in tokens]
    assert "is" not in [t.lower() for t in tokens]
    assert "the" not in [t.lower() for t in tokens]
    assert "minimum" in tokens
    assert "score" in tokens
    assert "trade" in tokens
    assert "system" in tokens


def test_tokenize_query_keeps_quoted_phrases_intact():
    tokens = _tokenize_query('Find the file called "trade gate config" please')
    assert "trade gate config" in tokens


def test_tokenize_query_preserves_identifiers_with_punctuation():
    tokens = _tokenize_query("What does BS_BRAIN_API_KEY do, and where is feature/vault-tools-v2 checked out?")
    assert "BS_BRAIN_API_KEY" in tokens
    assert "feature/vault-tools-v2" in tokens


def test_search_by_tokens_and_gate_prefers_full_match(tmp_path, monkeypatch):
    import obsidian_vault_mcp.config as config
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "RETRIEVAL_EXCLUDED_DIRS", set())

    (tmp_path / "full.md").write_text("alpha beta gamma delta\n")
    (tmp_path / "partial.md").write_text("alpha beta gamma\n")

    matches = _search_by_tokens(
        ["alpha", "beta", "gamma", "delta"], tmp_path, "*.md", 20, 1, require_all=True
    )
    paths = [m["path"] for m in matches]
    assert "full.md" in paths
    assert "partial.md" not in paths  # strict AND gate: missing "delta"


def test_search_by_tokens_allow_partial_recovers_near_miss(tmp_path, monkeypatch):
    """Core vault-retrieval-candidate-recall-v1 regression test: a small file
    missing one of many tokens must still surface as a lower-ranked candidate
    instead of being invisible to the keyword leg, once allow_partial=True."""
    import obsidian_vault_mcp.config as config
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "RETRIEVAL_EXCLUDED_DIRS", set())

    (tmp_path / "correct-but-partial.md").write_text(
        "The trade gate score floor is 48, configured via RISK_MIN_SCORE_TO_TRADE.\n"
    )
    # A large, unrelated file that happens to contain every token somewhere.
    huge_unrelated = " ".join(
        ["filler"] * 50 + ["trade", "gate", "score", "floor", "config", "knob", "number", "rejected", "outright", "setup"]
    )
    (tmp_path / "huge-changelog.md").write_text(huge_unrelated + "\n")

    tokens = ["trade", "gate", "score", "floor", "config", "knob", "number", "rejected", "outright", "setup"]

    without_partial = _search_by_tokens(tokens, tmp_path, "*.md", 20, 1, require_all=True, allow_partial=False)
    paths_without = {m["path"] for m in without_partial}
    assert "correct-but-partial.md" not in paths_without  # confirms the diagnosed bug reproduces

    with_partial = _search_by_tokens(tokens, tmp_path, "*.md", 20, 1, require_all=True, allow_partial=True)
    paths_with = {m["path"] for m in with_partial}
    assert "correct-but-partial.md" in paths_with  # now recoverable as a candidate
    assert "huge-changelog.md" in paths_with  # AND match still present, not displaced


def test_search_by_tokens_allow_partial_still_ranks_and_match_first(tmp_path, monkeypatch):
    import obsidian_vault_mcp.config as config
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "RETRIEVAL_EXCLUDED_DIRS", set())

    (tmp_path / "full.md").write_text("alpha beta gamma delta\n")
    (tmp_path / "partial.md").write_text("alpha beta gamma\n")

    matches = _search_by_tokens(
        ["alpha", "beta", "gamma", "delta"], tmp_path, "*.md", 20, 1,
        require_all=True, allow_partial=True,
    )
    seen_order = []
    for m in matches:
        if m["path"] not in seen_order:
            seen_order.append(m["path"])
    assert seen_order.index("full.md") < seen_order.index("partial.md")


def test_search_by_tokens_partial_respects_min_overlap(tmp_path, monkeypatch):
    """A file matching only a single token should not be promoted as a partial
    candidate -- _PARTIAL_MATCH_MIN_OVERLAP guards against pure noise matches."""
    import obsidian_vault_mcp.config as config
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "RETRIEVAL_EXCLUDED_DIRS", set())

    (tmp_path / "full.md").write_text("alpha beta gamma delta\n")
    (tmp_path / "one-token-only.md").write_text("alpha\n")

    matches = _search_by_tokens(
        ["alpha", "beta", "gamma", "delta"], tmp_path, "*.md", 20, 1,
        require_all=True, allow_partial=True,
    )
    paths = {m["path"] for m in matches}
    assert "one-token-only.md" not in paths


def test_search_by_tokens_allow_partial_default_false_unchanged():
    """Kill-switch check: allow_partial defaults to False, so any existing
    caller that doesn't pass it gets byte-identical pre-change behaviour."""
    import inspect
    sig = inspect.signature(_search_by_tokens)
    assert sig.parameters["allow_partial"].default is False


def test_search_keyword_fallback_unaffected_by_allow_partial_addition(tmp_path, monkeypatch):
    """_search_keyword_fallback never passes allow_partial -- confirms its
    call site wasn't accidentally changed."""
    import obsidian_vault_mcp.config as config
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "RETRIEVAL_EXCLUDED_DIRS", set())

    (tmp_path / "note.md").write_text("hello world\n")
    matches = _search_keyword_fallback("hello world", tmp_path, "*.md", 20, 1)
    assert any(m["path"] == "note.md" for m in matches)


def test_vault_search_tokenized_augmentation_appends_without_reordering_literal(tmp_path, monkeypatch):
    import obsidian_vault_mcp.config as config
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "RETRIEVAL_EXCLUDED_DIRS", set())
    monkeypatch.setattr(config, "VAULT_SEARCH_TOKENIZE", True)

    (tmp_path / "literal.md").write_text("this exact long natural language question phrase appears here verbatim\n")
    (tmp_path / "tokenized-only.md").write_text("exact long natural language phrase words present separately\n")

    result = json.loads(vault_search("this exact long natural language question phrase appears here verbatim"))
    assert result["results"][0]["path"] == "literal.md"
    assert result["results"][0]["match_type"] == "literal"


def test_augment_with_tokenized_matches_tags_literal_and_tokenized(tmp_path, monkeypatch):
    import obsidian_vault_mcp.config as config
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "RETRIEVAL_EXCLUDED_DIRS", set())

    (tmp_path / "tokenized.md").write_text("alpha beta gamma delta epsilon zeta eta theta\n")

    result = _augment_with_tokenized_matches(
        "alpha beta gamma delta epsilon zeta eta theta what",
        [], tmp_path, "*.md", 20, 1,
    )
    assert any(m["match_type"] == "tokenized" and m["path"] == "tokenized.md" for m in result)
