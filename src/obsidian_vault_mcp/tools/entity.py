"""Zero-LLM entity lookup for the Obsidian vault MCP server.

Reads `_entities.json` at the vault root, built nightly by scripts/dreaming.py
(a sibling CLI/cron repo) -- this tool only ever reads that file, never
rebuilds it. The JSON file is the entire interface between the two: no shared
code, no import between this package and dreaming.py.
"""

import difflib
import json
import logging

from .. import config
from ..utils import sanitize_for_json, SafeJSONEncoder
from ..vault import read_file

logger = logging.getLogger(__name__)

ENTITIES_FILE = "_entities.json"


def _load_entities() -> list[dict]:
    path = config.VAULT_PATH / ENTITIES_FILE
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("entities", [])
    except Exception as e:
        logger.error("vault_entity: failed to load %s: %s", path, e)
        return []


def _searchable_names(entity: dict) -> list[str]:
    return [entity["name"], *entity.get("aliases", [])]


def _tokenize(s: str) -> set[str]:
    return {t for t in s.lower().replace(",", " ").replace("&", " ").split() if t}


def _match_entities(query: str, entities: list[dict]) -> tuple[list[dict], str]:
    """Match a query against entity canonical names + aliases.

    Priority: exact -> substring -> token-subset -> fuzzy.
    Returns (matched_entities, match_type); match_type is one of
    'exact', 'substring', 'token', 'fuzzy', 'none'.
    """
    query_lower = query.strip().lower()
    if not query_lower:
        return [], "none"

    exact = [e for e in entities if query_lower in {n.lower() for n in _searchable_names(e)}]
    if exact:
        return exact, "exact"

    substring = [e for e in entities if any(query_lower in n.lower() for n in _searchable_names(e))]
    if substring:
        return substring, "substring"

    query_tokens = _tokenize(query)
    token_matches = [
        e for e in entities if any(query_tokens <= _tokenize(n) for n in _searchable_names(e))
    ]
    if token_matches:
        return token_matches, "token"

    all_names = [n for e in entities for n in _searchable_names(e)]
    close = difflib.get_close_matches(query, all_names, n=5, cutoff=0.6)
    if close:
        close_lower = {c.lower() for c in close}
        fuzzy = [e for e in entities if any(n.lower() in close_lower for n in _searchable_names(e))]
        return fuzzy, "fuzzy"

    return [], "none"


def vault_entity(name: str, max_backlinks: int = 15) -> str:
    """Look up a vault entity by name/alias. Single match returns the entity's
    page + backlinks; multiple matches return disambiguation candidates; no
    match returns nearest names."""
    entities = _load_entities()
    result: dict = {}

    if not entities:
        result["entity"] = None
        result["message"] = (
            f"No {ENTITIES_FILE} found at vault root -- run the nightly "
            "dreaming cycle to build the entity index."
        )
        return json.dumps(sanitize_for_json(result), cls=SafeJSONEncoder)

    matches, match_type = _match_entities(name, entities)

    if not matches:
        all_names = [n for e in entities for n in _searchable_names(e)]
        result["entity"] = None
        result["nearest"] = difflib.get_close_matches(name, all_names, n=5, cutoff=0.3)
        result["message"] = f"No entity found matching '{name}'."
    elif len(matches) > 1:
        result["entity"] = None
        result["candidates"] = [
            {"name": e["name"], "path": e["path"], "type": e.get("type")} for e in matches
        ]
        result["message"] = (
            f"Ambiguous match for '{name}': {len(matches)} candidates. "
            "Provide more of the name to disambiguate."
        )
    else:
        entity = matches[0]
        rel_path = entity["path"]
        try:
            content, _ = read_file(rel_path)
        except Exception as e:
            logger.error("vault_entity: error reading %s: %s", rel_path, e)
            content = None

        backlinks = entity.get("backlinks", [])
        result["entity"] = {
            "name": entity["name"],
            "path": rel_path,
            "type": entity.get("type"),
            "aliases": entity.get("aliases", []),
            "content": content,
            "backlinks": backlinks[:max_backlinks],
            "backlink_count_total": len(backlinks),
            "backlinks_truncated_at_source": entity.get("backlinks_truncated", False),
        }

    return json.dumps(sanitize_for_json(result), cls=SafeJSONEncoder)
