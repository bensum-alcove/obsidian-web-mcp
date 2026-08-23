"""Tests for tools/query.py's entity-mention candidate leg
(vault-retrieval-entity-resolution-r5-v2) -- _entity_leg and the
_entities.json-derived token index it's built on."""

import json

import pytest

from obsidian_vault_mcp import config
from obsidian_vault_mcp.tools import query as query_tool


def _write_entities(vault, entities):
    (vault / "_entities.json").write_text(
        json.dumps({"vault": "bs-brain", "generated": "2026-08-23T00:00:00Z",
                    "entity_count": len(entities), "entities": entities}),
        encoding="utf-8",
    )


def _entity(name, path, type_="client", aliases=None, descriptors=None):
    return {
        "name": name, "path": path, "type": type_,
        "aliases": aliases or [], "descriptors": descriptors or [],
        "backlinks": [], "backlinks_truncated": False,
    }


# --- _entity_query_keys -------------------------------------------------------

def test_entity_query_keys_splits_slash_joined_names():
    keys = query_tool._entity_query_keys("the Eqbal/Ucas refinance discount")
    assert "eqbal" in keys
    assert "ucas" in keys


def test_entity_query_keys_splits_possessive_suffix():
    keys = query_tool._entity_query_keys("her partner Vasey's investment property")
    assert "vasey" in keys


def test_entity_query_keys_drops_stopwords_and_interrogatives():
    keys = query_tool._entity_query_keys("What is the plan for the household?")
    assert "what" not in keys
    assert "the" not in keys


# --- token extraction ---------------------------------------------------------

def test_entity_surname_tokens_from_canonical_name_and_aliases():
    entity = _entity(
        "Sangster, Tiffany & Vasey, Lucy",
        "Clients/Sangster, Tiffany & Vasey, Lucy.md",
        aliases=["Tiffany Sangster", "Lucy Vasey"],
    )
    tokens = query_tool._entity_surname_tokens(entity)
    assert tokens == {"sangster", "vasey"}


def test_entity_surname_tokens_below_min_length_dropped():
    entity = _entity("Ng, Bo", "Clients/Ng, Bo.md", aliases=["Bo Ng"])
    assert query_tool._entity_surname_tokens(entity) == set()


def test_entity_given_name_tokens_from_canonical_name_and_aliases():
    entity = _entity(
        "Eqbal, Yusuf", "Clients/Eqbal, Yusuf.md", aliases=["Yusuf Eqbal", "Jasmine Ucas"],
    )
    tokens = query_tool._entity_given_name_tokens(entity)
    assert tokens == {"yusuf", "jasmine"}


def test_entity_descriptor_tokens_respects_min_length():
    entity = _entity(
        "Eqbal, Yusuf", "Clients/Eqbal, Yusuf.md",
        descriptors=["Nephrologist, QLD Health (Medical SMO)"],
    )
    tokens = query_tool._entity_descriptor_tokens(entity)
    assert "nephrologist" in tokens
    assert "health" in tokens
    # "qld" (3 chars) and "smo" (3 chars) fall below _DESCRIPTOR_TOKEN_MIN_LEN
    assert "qld" not in tokens


# --- _load_entity_token_index -------------------------------------------------

def test_load_entity_token_index_excludes_reference_type(vault_dir, monkeypatch):
    """Reference docs (specs, canonical-state records) don't participate in
    this leg -- see the module-level rationale comment in query.py for the
    frozen-v3 canonical_contradiction_precedence regression this prevents."""
    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    monkeypatch.setattr(query_tool, "_entity_token_index_cache", None)
    _write_entities(vault_dir, [
        _entity("SYSTEM-FACTS", "SYSTEM-FACTS.md", type_="reference", aliases=["System Facts"]),
        _entity("Eqbal, Yusuf", "Clients/Eqbal, Yusuf.md", aliases=["Yusuf Eqbal"]),
    ])
    index = query_tool._load_entity_token_index()
    assert "facts" not in index
    assert "eqbal" in index


def test_load_entity_token_index_caps_common_descriptor_words(vault_dir, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    monkeypatch.setattr(query_tool, "_entity_token_index_cache", None)
    _write_entities(vault_dir, [
        _entity("A, One", "a.md", aliases=["One A"], descriptors=["Analyst, current employer"]),
        _entity("B, Two", "b.md", aliases=["Two B"], descriptors=["Manager, current employer"]),
        _entity("C, Three", "c.md", aliases=["Three C"], descriptors=["Trader, current employer"]),
    ])
    index = query_tool._load_entity_token_index()
    # "current" is shared by all 3 entities -- over the cap, dropped entirely
    assert "current" not in index
    # but each entity's own distinctive occupation word survives
    assert "analyst" in index
    assert "manager" in index
    assert "trader" in index


def test_load_entity_token_index_caps_common_given_names(vault_dir, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    monkeypatch.setattr(query_tool, "_entity_token_index_cache", None)
    _write_entities(vault_dir, [
        _entity("Aaa, Adam", "a.md", aliases=["Adam Aaa"]),
        _entity("Bbb, Adam", "b.md", aliases=["Adam Bbb"]),
        _entity("Ccc, Adam", "c.md", aliases=["Adam Ccc"]),
        _entity("Sangster, Tiffany & Vasey, Lucy", "d.md", aliases=["Tiffany Sangster", "Lucy Vasey"]),
    ])
    index = query_tool._load_entity_token_index()
    # "adam" shared by 3 different clients -- not distinctive, dropped
    assert "adam" not in index
    # "lucy" shared by exactly one client, at/under the cap -- kept
    assert "lucy" in index
    assert index["lucy"] == ["d.md"]


def test_load_entity_token_index_cache_invalidated_by_mtime(vault_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    monkeypatch.setattr(query_tool, "_entity_token_index_cache", None)
    _write_entities(vault_dir, [_entity("Eqbal, Yusuf", "Clients/Eqbal, Yusuf.md", aliases=["Yusuf Eqbal"])])
    first = query_tool._load_entity_token_index()
    assert "eqbal" in first

    import time
    time.sleep(0.01)
    _write_entities(vault_dir, [_entity("Vasey, Lucy", "Clients/Vasey, Lucy.md", aliases=["Lucy Vasey"])])
    second = query_tool._load_entity_token_index()
    assert "vasey" in second
    assert "eqbal" not in second


def test_load_entity_token_index_missing_file_returns_empty(vault_dir, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    monkeypatch.setattr(query_tool, "_entity_token_index_cache", None)
    assert query_tool._load_entity_token_index() == {}


# --- _entity_leg ---------------------------------------------------------------

def test_entity_leg_matches_surname_buried_in_descriptive_sentence(vault_dir, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    monkeypatch.setattr(query_tool, "_entity_token_index_cache", None)
    _write_entities(vault_dir, [
        _entity(
            "Sangster, Tiffany & Vasey, Lucy", "Clients/Sangster, Tiffany & Vasey, Lucy.md",
            aliases=["Tiffany Sangster", "Lucy Vasey"],
        ),
    ])
    result = query_tool._entity_leg("her partner Vasey's investment property", 8)
    assert result == ["Clients/Sangster, Tiffany & Vasey, Lucy.md"]


def test_entity_leg_matches_occupation_descriptor_with_no_name_at_all(vault_dir, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    monkeypatch.setattr(query_tool, "_entity_token_index_cache", None)
    _write_entities(vault_dir, [
        _entity(
            "Asimus, Angie", "Clients/Asimus, Angie.md", aliases=["Angie Asimus"],
            descriptors=["Weekend Co-Host & Newsreader, Seven Network"],
        ),
    ])
    result = query_tool._entity_leg("a TV newsreader with a farming side income", 8)
    assert result == ["Clients/Asimus, Angie.md"]


def test_entity_leg_ambiguous_surname_returns_all_candidates_not_one(vault_dir, monkeypatch):
    """Two unrelated households sharing a surname must both surface -- this leg
    never collapses ambiguity into a single forced answer."""
    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    monkeypatch.setattr(query_tool, "_entity_token_index_cache", None)
    _write_entities(vault_dir, [
        _entity("McGrath, Danny", "a.md", aliases=["Danny McGrath"]),
        _entity("Robson, Lloyd & McGrath, Rebecca", "b.md", aliases=["Lloyd Robson", "Rebecca McGrath"]),
    ])
    result = query_tool._entity_leg("checking in on the McGrath file", 8)
    assert set(result) == {"a.md", "b.md"}


def test_entity_leg_ranks_by_distinct_token_matches(vault_dir, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    monkeypatch.setattr(query_tool, "_entity_token_index_cache", None)
    _write_entities(vault_dir, [
        _entity("Eqbal, Yusuf", "eqbal.md", aliases=["Yusuf Eqbal", "Jasmine Ucas"]),
        _entity("Holloway, Jon", "holloway.md", aliases=["Jon Holloway"]),
    ])
    # matches both "eqbal" and "ucas" for the first entity, only nothing for the second
    result = query_tool._entity_leg("the Eqbal/Ucas refinance discount", 8)
    assert result[0] == "eqbal.md"


def test_entity_leg_respects_max_candidates_cap(vault_dir, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    monkeypatch.setattr(query_tool, "_entity_token_index_cache", None)
    entities = [
        _entity(f"Surname{i}, Given{i}", f"c{i}.md", aliases=[f"Given{i} Surname{i}"])
        for i in range(10)
    ]
    _write_entities(vault_dir, entities)
    query = " ".join(f"Surname{i}" for i in range(10))
    result = query_tool._entity_leg(query, 3)
    assert len(result) == 3


def test_entity_leg_no_match_returns_empty(vault_dir, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    monkeypatch.setattr(query_tool, "_entity_token_index_cache", None)
    _write_entities(vault_dir, [_entity("Eqbal, Yusuf", "eqbal.md", aliases=["Yusuf Eqbal"])])
    assert query_tool._entity_leg("what time does the nightly backup run", 8) == []


# --- _rrf_fuse three-leg behaviour --------------------------------------------

def test_rrf_fuse_entity_leg_is_additive_third_leg():
    scores = query_tool._rrf_fuse(["a.md"], ["b.md"], ["a.md"], k=6)
    # a.md appears in both keyword (rank 1) and entity (rank 1) legs
    assert scores["a.md"] == pytest.approx(1 / 7 + 1 / 7)
    assert scores["b.md"] == pytest.approx(1 / 7)


def test_rrf_fuse_backward_compatible_without_entity_paths():
    scores = query_tool._rrf_fuse(["a.md"], ["b.md"], k=6)
    assert scores == {"a.md": pytest.approx(1 / 7), "b.md": pytest.approx(1 / 7)}


# --- vault_query integration ---------------------------------------------------

def test_vault_query_entity_leg_surfaces_file_with_no_lexical_overlap(vault_dir, monkeypatch):
    """End-to-end: a client file with zero literal keyword overlap against the
    query, and no semantic index built in this test process, is still
    surfaced -- purely through the entity-mention leg."""
    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    monkeypatch.setattr(config, "RETRIEVAL_EXCLUDED_DIRS", set())
    monkeypatch.setattr(query_tool, "_entity_token_index_cache", None)
    (vault_dir / "Clients").mkdir()
    (vault_dir / "Clients" / "Sangster, Tiffany & Vasey, Lucy.md").write_text(
        "---\ntype: client\n---\n\nInvestment unit in Beaconsfield, LVR 67.72%.\n"
    )
    _write_entities(vault_dir, [
        _entity(
            "Sangster, Tiffany & Vasey, Lucy", "Clients/Sangster, Tiffany & Vasey, Lucy.md",
            aliases=["Tiffany Sangster", "Lucy Vasey"],
        ),
    ])
    result = json.loads(query_tool.vault_query("her partner Vasey's investment property loan"))
    paths = [r["path"] for r in result["results"]]
    assert "Clients/Sangster, Tiffany & Vasey, Lucy.md" in paths


def test_vault_query_entity_expansion_kill_switch_disables_leg(vault_dir, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    monkeypatch.setattr(config, "RETRIEVAL_EXCLUDED_DIRS", set())
    monkeypatch.setattr(config, "VAULT_QUERY_ENTITY_EXPANSION", False)
    monkeypatch.setattr(query_tool, "_entity_token_index_cache", None)
    (vault_dir / "Clients").mkdir()
    # Deliberately zero lexical overlap with the query below (besides the
    # surname itself) -- isolates the entity leg's contribution from the
    # keyword leg's, which the earlier positive test doesn't need to isolate.
    (vault_dir / "Clients" / "Sangster, Tiffany & Vasey, Lucy.md").write_text(
        "---\ntype: client\n---\n\nConfidential household file, internal notes only.\n"
    )
    _write_entities(vault_dir, [
        _entity(
            "Sangster, Tiffany & Vasey, Lucy", "Clients/Sangster, Tiffany & Vasey, Lucy.md",
            aliases=["Tiffany Sangster", "Lucy Vasey"],
        ),
    ])
    result = json.loads(query_tool.vault_query("her partner Vasey's situation"))
    paths = [r["path"] for r in result["results"]]
    assert "Clients/Sangster, Tiffany & Vasey, Lucy.md" not in paths
