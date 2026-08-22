"""Tests for vault_semantic_search chunking mechanisms added in
vault-retrieval-candidate-recall-v1: frontmatter stripping/prefixing and
per-table-row sub-chunking. See tools/semantic_search.py for the diagnosed
failure mode (dense reference tables diluting chunk-level embeddings)."""

import obsidian_vault_mcp.config as config
from obsidian_vault_mcp.tools.semantic_search import (
    _strip_frontmatter,
    _frontmatter_prefix_text,
    _split_table_rows,
    _chunk_text,
    _prepare_content_for_chunking,
)


# --- frontmatter stripping -------------------------------------------------

def test_strip_frontmatter_removes_yaml_block():
    content = "---\ntype: canonical-state\ncomponent_id: foo\n---\n\n# Foo\n\nBody text.\n"
    body, fm = _strip_frontmatter(content)
    assert "---" not in body
    assert "component_id" not in body
    assert "Body text." in body
    assert fm["type"] == "canonical-state"
    assert fm["component_id"] == "foo"


def test_strip_frontmatter_no_frontmatter_is_noop():
    content = "# Just a heading\n\nNo frontmatter here.\n"
    body, fm = _strip_frontmatter(content)
    assert body == content
    assert fm == {}


def test_strip_frontmatter_malformed_yaml_does_not_raise():
    content = "---\n: : : broken yaml [\n---\n\nBody.\n"
    body, fm = _strip_frontmatter(content)
    assert isinstance(fm, dict)


def test_frontmatter_prefix_text_uses_allowlisted_fields():
    prefix = _frontmatter_prefix_text({
        "component_id": "brain-dashboard",
        "state": "active",
        "revision": "sha256:deadbeef",  # not in allowlist -- must be excluded
    })
    assert "brain-dashboard" in prefix
    assert "active" in prefix
    assert "deadbeef" not in prefix


def test_frontmatter_prefix_text_empty_when_no_allowlisted_fields():
    assert _frontmatter_prefix_text({"revision": "sha256:x", "updated": "2026-08-01"}) == ""


def test_prepare_content_for_chunking_strips_by_default(monkeypatch):
    monkeypatch.setattr(config, "VAULT_SEMANTIC_STRIP_FRONTMATTER", True)
    content = "---\ntype: canonical-state\n---\n\nBody.\n"
    body, fm = _prepare_content_for_chunking(content)
    assert "type: canonical-state" not in body
    assert fm["type"] == "canonical-state"


def test_prepare_content_for_chunking_kill_switch_reproduces_old_behaviour(monkeypatch):
    """Kill switch off must be byte-identical to pre-change behaviour: frontmatter
    left inline, no separate dict extracted."""
    monkeypatch.setattr(config, "VAULT_SEMANTIC_STRIP_FRONTMATTER", False)
    content = "---\ntype: canonical-state\n---\n\nBody.\n"
    body, fm = _prepare_content_for_chunking(content)
    assert body == content
    assert fm == {}


def test_chunk_text_prepends_frontmatter_prefix_to_first_chunk(monkeypatch):
    monkeypatch.setattr(config, "VAULT_SEMANTIC_FRONTMATTER_PREFIX", True)
    monkeypatch.setattr(config, "VAULT_SEMANTIC_TABLE_ROW_CHUNKING", True)
    body = "Public dashboard at brain.bensum.org, backend on port 8432.\n"
    chunks = _chunk_text(body, {"component_id": "brain-dashboard", "state": "active"})
    assert len(chunks) >= 1
    assert "brain-dashboard" in chunks[0]["content"]
    assert "8432" in chunks[0]["content"]  # original content preserved, not replaced


def test_chunk_text_frontmatter_prefix_kill_switch(monkeypatch):
    monkeypatch.setattr(config, "VAULT_SEMANTIC_FRONTMATTER_PREFIX", False)
    monkeypatch.setattr(config, "VAULT_SEMANTIC_TABLE_ROW_CHUNKING", True)
    body = "Some body text.\n"
    chunks = _chunk_text(body, {"component_id": "brain-dashboard"})
    assert "brain-dashboard" not in chunks[0]["content"]


# --- table-row sub-chunking -------------------------------------------------

def test_split_table_rows_returns_none_for_non_table_text():
    text = "Just a paragraph of prose with no table structure at all.\n\nAnother paragraph."
    assert _split_table_rows(text) is None


def test_split_table_rows_returns_none_for_small_tables():
    text = (
        "| Fact | Value |\n"
        "|------|-------|\n"
        "| A | 1 |\n"
        "| B | 2 |\n"
    )
    # Below _MIN_TABLE_ROWS_TO_SPLIT (4) -- not worth splitting.
    assert _split_table_rows(text) is None


def test_split_table_rows_splits_dense_fact_table():
    text = (
        "| Fact | Value |\n"
        "|------|-------|\n"
        "| SPI dashboard | localhost:8000 |\n"
        "| HSI dashboard | localhost:8002 |\n"
        "| IBKR client IDs | SPI: 3, HSI: 6 |\n"
        "| Trade gate | MIN_SCORE_TO_TRADE = 48, in risk/config.py |\n"
    )
    entries = _split_table_rows(text)
    assert entries is not None
    assert len(entries) == 4
    trade_gate_entry = [e for e in entries if "MIN_SCORE_TO_TRADE" in e][0]
    assert "Trade gate" in trade_gate_entry
    assert "48" in trade_gate_entry
    # Each row is now independent -- the SPI dashboard fact must not appear
    # in the trade-gate row's own chunk text (this is the dilution fix).
    assert "SPI dashboard" not in trade_gate_entry


def test_split_table_rows_preserves_leading_prose():
    text = (
        "Some introductory sentence before the table.\n\n"
        "| Fact | Value |\n"
        "|------|-------|\n"
        "| A | 1 |\n"
        "| B | 2 |\n"
        "| C | 3 |\n"
        "| D | 4 |\n"
    )
    entries = _split_table_rows(text)
    assert entries is not None
    assert any("introductory sentence" in e for e in entries)


def test_chunk_text_table_row_chunking_produces_multiple_chunks(monkeypatch):
    monkeypatch.setattr(config, "VAULT_SEMANTIC_TABLE_ROW_CHUNKING", True)
    monkeypatch.setattr(config, "VAULT_SEMANTIC_FRONTMATTER_PREFIX", False)
    body = (
        "## Edge Trading System\n\n"
        "| Fact | Value |\n"
        "|------|-------|\n"
        "| SPI dashboard | localhost:8000 |\n"
        "| HSI dashboard | localhost:8002 |\n"
        "| IBKR client IDs | SPI: 3, HSI: 6 |\n"
        "| Trade gate | MIN_SCORE_TO_TRADE = 48, in risk/config.py |\n"
    )
    chunks = _chunk_text(body, {})
    assert len(chunks) == 4
    assert all(c["heading"] == "Edge Trading System" for c in chunks)
    trade_gate_chunk = [c for c in chunks if "MIN_SCORE_TO_TRADE" in c["content"]][0]
    assert "SPI dashboard" not in trade_gate_chunk["content"]


def test_chunk_text_table_row_chunking_kill_switch(monkeypatch):
    """Kill switch off reproduces the previous whole-section chunk (one chunk
    for the entire table, diluted embedding included)."""
    monkeypatch.setattr(config, "VAULT_SEMANTIC_TABLE_ROW_CHUNKING", False)
    monkeypatch.setattr(config, "VAULT_SEMANTIC_FRONTMATTER_PREFIX", False)
    body = (
        "## Edge Trading System\n\n"
        "| Fact | Value |\n"
        "|------|-------|\n"
        "| SPI dashboard | localhost:8000 |\n"
        "| HSI dashboard | localhost:8002 |\n"
        "| IBKR client IDs | SPI: 3, HSI: 6 |\n"
        "| Trade gate | MIN_SCORE_TO_TRADE = 48, in risk/config.py |\n"
    )
    chunks = _chunk_text(body, {})
    assert len(chunks) == 1
    assert "SPI dashboard" in chunks[0]["content"]
    assert "MIN_SCORE_TO_TRADE" in chunks[0]["content"]


def test_chunk_text_non_table_sections_unaffected():
    """Regression guard: ordinary prose sections must chunk exactly as before,
    regardless of table-row-chunking flag state."""
    body = (
        "## Some Section\n\n"
        "This is a normal paragraph of prose with no table in it whatsoever, "
        "just regular explanatory sentences about the system.\n"
    )
    chunks_on = _chunk_text(body, {})
    assert len(chunks_on) == 1
    assert "normal paragraph" in chunks_on[0]["content"]
