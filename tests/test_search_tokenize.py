"""Tests for the vault_query keyword-leg tokenizer and shared per-token search
(tools/search.py: _tokenize_query, _search_by_tokens). See tools/query.py's
_keyword_leg for how these compose into the tokenized keyword leg."""

from obsidian_vault_mcp import config
from obsidian_vault_mcp.tools.search import _tokenize_query, _search_by_tokens


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
