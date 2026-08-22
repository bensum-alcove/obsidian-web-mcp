"""Tests for the vault_query keyword-leg tokenizer and shared per-token search
(tools/search.py: _tokenize_query, _search_by_tokens). See tools/query.py's
_keyword_leg for how these compose into the tokenized keyword leg."""

import json

from obsidian_vault_mcp import config
from obsidian_vault_mcp.tools.search import (
    _tokenize_query,
    _search_by_tokens,
    _search_keyword_fallback,
    _augment_with_tokenized_matches,
    vault_search,
)


# --- _tokenize_query -----------------------------------------------------------

def test_tokenize_query_strips_stopwords_and_interrogatives():
    tokens = [t.lower() for t in _tokenize_query("What is the port for the vault mcp server?")]
    for stop in ("what", "is", "the", "for"):
        assert stop not in tokens
    for content in ("port", "vault", "mcp", "server"):
        assert content in tokens


def test_tokenize_query_retains_literal_identifier_in_full_sentence():
    tokens = _tokenize_query(
        "Does a BS_BRAIN_API_KEY environment variable exist anywhere in the infrastructure?"
    )
    assert "BS_BRAIN_API_KEY" in tokens


def test_tokenize_query_does_not_split_underscores():
    tokens = _tokenize_query("look up BS_BRAIN_API_KEY now")
    assert "BS_BRAIN_API_KEY" in tokens
    assert "BS" not in tokens
    assert "BRAIN" not in tokens
    assert "API" not in tokens
    assert "KEY" not in tokens


def test_tokenize_query_does_not_split_slashes_or_hyphens():
    tokens = _tokenize_query("which branch is feature/vault-tools-v2 based on")
    assert "feature/vault-tools-v2" in tokens


def test_tokenize_query_does_not_split_dots_or_leading_tilde():
    tokens = _tokenize_query("where does ~/.config/supervisor/ live on disk")
    assert "~/.config/supervisor/" in tokens


def test_tokenize_query_preserves_bare_numeric_error_code():
    tokens = _tokenize_query("what does a 502 from vault mcp mean")
    assert "502" in tokens


def test_tokenize_query_preserves_quoted_phrase_intact():
    tokens = _tokenize_query('find the note about "cease and desist" letter')
    assert "cease and desist" in tokens
    assert "cease" not in tokens
    assert "desist" not in tokens


def test_tokenize_query_strips_trailing_sentence_punctuation():
    tokens = _tokenize_query("does infrastructure mention this, exactly?")
    assert "exactly" in tokens
    assert "exactly?" not in tokens
    assert "this," not in tokens


def test_tokenize_query_stopword_only_query_returns_empty_no_crash():
    assert _tokenize_query("what is the of a") == []


def test_tokenize_query_short_all_content_query():
    assert _tokenize_query("BS_BRAIN_API_KEY") == ["BS_BRAIN_API_KEY"]


# --- _search_by_tokens -----------------------------------------------------------

def test_search_by_tokens_empty_keywords_returns_empty(vault_dir):
    assert _search_by_tokens([], config.VAULT_PATH, "*.md", 10, 1) == []


def test_search_by_tokens_and_logic_prefers_file_matching_all_keywords(vault_dir):
    (vault_dir / "strong.md").write_text("alpha beta gamma content here\n")
    (vault_dir / "weak.md").write_text("alpha content only\n")

    matches = _search_by_tokens(["alpha", "beta", "gamma"], config.VAULT_PATH, "*.md", 10, 1)
    paths_in_order = []
    for m in matches:
        if m["path"] not in paths_in_order:
            paths_in_order.append(m["path"])

    # AND logic: a file matching every keyword excludes weaker single-keyword files.
    assert paths_in_order == ["strong.md"]


def test_search_by_tokens_require_all_false_keeps_partial_match_alive(vault_dir):
    """With require_all=True (the default, matching _search_keyword_fallback), a
    file matching every keyword excludes a partial match entirely. With
    require_all=False, the partial match survives (ranked lower)."""
    (vault_dir / "full-match.md").write_text("alpha beta gamma\n")
    (vault_dir / "partial-match.md").write_text("alpha only\n")

    require_all_true = _search_by_tokens(
        ["alpha", "beta", "gamma"], config.VAULT_PATH, "*.md", 10, 1, require_all=True
    )
    paths_true = {m["path"] for m in require_all_true}
    assert paths_true == {"full-match.md"}

    require_all_false = _search_by_tokens(
        ["alpha", "beta", "gamma"], config.VAULT_PATH, "*.md", 10, 1, require_all=False
    )
    paths_false = {m["path"] for m in require_all_false}
    assert paths_false == {"full-match.md", "partial-match.md"}


def test_search_by_tokens_ranks_more_distinct_matches_higher_when_no_and_match(vault_dir):
    # No file matches all three keywords, so OR logic ranks by distinct-token count.
    (vault_dir / "two-tokens.md").write_text("alpha beta content here\n")
    (vault_dir / "one-token.md").write_text("alpha content only\n")

    matches = _search_by_tokens(["alpha", "beta", "gamma"], config.VAULT_PATH, "*.md", 10, 1)
    paths_in_order = []
    for m in matches:
        if m["path"] not in paths_in_order:
            paths_in_order.append(m["path"])

    assert paths_in_order[0] == "two-tokens.md"
    assert "one-token.md" in paths_in_order


# --- vault-retrieval-candidate-recall-v1 unique tests (dev checkout) ---------
# Ported in during checkout reconciliation (vault-retrieval-r5-085-v1). These
# pin the allow_partial keyword-candidate mechanism that the live checkout's
# suite above does not cover.



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
