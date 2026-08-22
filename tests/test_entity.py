"""Tests for tools/entity.py -- the vault_entity zero-LLM lookup tool."""

import json

import pytest

from obsidian_vault_mcp import config
from obsidian_vault_mcp.tools import entity as entity_tool


def _write_entities(vault, entities):
    (vault / "_entities.json").write_text(
        json.dumps({"vault": "bs-brain", "generated": "2026-07-10T00:00:00Z",
                    "entity_count": len(entities), "entities": entities}),
        encoding="utf-8",
    )


@pytest.fixture
def entity_vault(vault_dir, monkeypatch):
    (vault_dir / "McGrath, Danny.md").write_text(
        "---\ntype: client\n---\n\n# McGrath, Danny\n\nRefinance in progress.\n"
    )
    (vault_dir / "McGrath, Michael & McGrath, Kim.md").write_text(
        "---\ntype: client\n---\n\n# McGrath, Michael & McGrath, Kim\n\nSettled.\n"
    )
    (vault_dir / "Robson, Lloyd & McGrath, Rebecca.md").write_text(
        "---\ntype: client\n---\n\n# Robson, Lloyd & McGrath, Rebecca\n"
    )
    (vault_dir / "Boyd, Nick.md").write_text("---\ntype: client\n---\n\n# Boyd, Nick\n")

    _write_entities(vault_dir, [
        {
            "name": "McGrath, Danny", "path": "McGrath, Danny.md", "type": "client",
            "aliases": ["Danny McGrath"],
            "backlinks": [{"path": "notes.md", "line": 3, "text": "Called [[McGrath, Danny]] today."}],
            "backlinks_truncated": False,
        },
        {
            "name": "McGrath, Michael & McGrath, Kim", "path": "McGrath, Michael & McGrath, Kim.md",
            "type": "client", "aliases": ["Michael McGrath", "Kim McGrath"],
            "backlinks": [], "backlinks_truncated": False,
        },
        {
            "name": "Robson, Lloyd & McGrath, Rebecca", "path": "Robson, Lloyd & McGrath, Rebecca.md",
            "type": "client", "aliases": ["Lloyd Robson", "Rebecca McGrath"],
            "backlinks": [], "backlinks_truncated": False,
        },
        {
            "name": "Boyd, Nick", "path": "Boyd, Nick.md", "type": "client",
            "aliases": ["Nick Boyd"],
            "backlinks": [{"path": f"mention-{i}.md", "line": 1, "text": "x"} for i in range(20)],
            "backlinks_truncated": False,
        },
    ])

    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    return vault_dir


def test_no_entities_json_returns_message(vault_dir, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", vault_dir)
    result = json.loads(entity_tool.vault_entity("Anyone"))
    assert result["entity"] is None
    assert "_entities.json" in result["message"]


def test_exact_match_returns_content_and_backlinks(entity_vault):
    result = json.loads(entity_tool.vault_entity("Boyd, Nick"))
    assert result["entity"]["name"] == "Boyd, Nick"
    assert "Boyd, Nick" in result["entity"]["content"]
    assert result["entity"]["backlink_count_total"] == 20


def test_alias_match(entity_vault):
    result = json.loads(entity_tool.vault_entity("Nick Boyd"))
    assert result["entity"]["name"] == "Boyd, Nick"


def test_max_backlinks_caps_returned_list(entity_vault):
    result = json.loads(entity_tool.vault_entity("Boyd, Nick", max_backlinks=5))
    assert len(result["entity"]["backlinks"]) == 5
    assert result["entity"]["backlink_count_total"] == 20


def test_mcgrath_disambiguates_multiple_candidates(entity_vault):
    result = json.loads(entity_tool.vault_entity("McGrath"))
    assert result["entity"] is None
    names = {c["name"] for c in result["candidates"]}
    assert names == {
        "McGrath, Danny",
        "McGrath, Michael & McGrath, Kim",
        "Robson, Lloyd & McGrath, Rebecca",
    }


def test_no_match_returns_nearest(entity_vault):
    result = json.loads(entity_tool.vault_entity("Zzyzx Nonexistent"))
    assert result["entity"] is None
    assert "nearest" in result
