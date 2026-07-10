"""Scoped session-context tools for the Obsidian vault MCP server."""

import difflib
import json
import logging
from pathlib import Path

import frontmatter as fm_lib

from .. import config
from ..utils import sanitize_for_json, SafeJSONEncoder
from ..vault import read_file

logger = logging.getLogger(__name__)

CLIENTS_DIR = "BS 2nd Brain/Alcove/Clients"
HOT_MD_PATH = "BS 2nd Brain/Alcove/Skills/hot.md"
INSTRUCTIONS_PATH = "BS 2nd Brain/Alcove/Skills/Lending-Analyst-Project-Instructions.md"


def _list_client_files() -> list[Path]:
    """Return sorted list of .md files in the Clients directory."""
    clients_abs = (config.VAULT_PATH / CLIENTS_DIR).resolve()
    if not clients_abs.is_dir():
        return []
    return sorted(clients_abs.glob("*.md"))


def _match_client(client: str, client_files: list[Path]) -> tuple[list[Path], str]:
    """Match a client query string against client file stems.

    Priority: exact-stem → surname-prefix → substring → fuzzy.
    Returns (matched_files, match_type).
    match_type is one of: 'exact', 'substring', 'fuzzy', 'none'.
    """
    client_lower = client.strip().lower()
    stems = {f: f.stem.lower() for f in client_files}

    # 1. Exact full-stem match
    exact = [f for f, stem in stems.items() if stem == client_lower]
    if exact:
        return exact, "exact"

    # 2. Surname-prefix: stem starts with "surname," or " surname,"
    prefix = [f for f, stem in stems.items() if stem.startswith(client_lower + ",")]
    if prefix:
        if len(prefix) == 1:
            return prefix, "substring"
        return prefix, "ambiguous"

    # 3. General substring match
    substring = [f for f, stem in stems.items() if client_lower in stem]
    if substring:
        if len(substring) == 1:
            return substring, "substring"
        return substring, "ambiguous"

    # 4. Fuzzy fallback
    all_stems = [f.stem for f in client_files]
    close_names = difflib.get_close_matches(client, all_stems, n=5, cutoff=0.6)
    if close_names:
        fuzzy = [f for f in client_files if f.stem in close_names]
        return fuzzy, "fuzzy"

    return [], "none"


def vault_client_context(
    client: str,
    include_hot: bool = True,
    include_instructions: bool = False,
) -> str:
    """Return scoped session context for a single client in one call.

    Replaces: vault_session_start + hot.md read + client search + client note read.
    """
    result: dict = {}

    # --- Client note ---
    client_files = _list_client_files()
    matches, match_type = _match_client(client, client_files)

    if not matches:
        result["client_note"] = None
        result["template_path"] = f"{CLIENTS_DIR}/{client}.md"
        result["message"] = (
            f"No client note found matching '{client}'. "
            f"Create one at {CLIENTS_DIR}/{client}.md."
        )
    elif match_type == "ambiguous" or (match_type == "fuzzy" and len(matches) > 1):
        result["client_note"] = None
        result["matches"] = [
            {"path": str(f.relative_to(config.VAULT_PATH)), "stem": f.stem}
            for f in matches
        ]
        result["message"] = (
            f"Ambiguous match for '{client}': {len(matches)} candidates. "
            "Provide more of the surname or add first name initials to disambiguate."
        )
    else:
        matched_file = matches[0]
        rel_path = str(matched_file.relative_to(config.VAULT_PATH))

        try:
            content, metadata = read_file(rel_path)
            fm_data: dict = {}
            try:
                post = fm_lib.loads(content)
                fm_data = dict(post.metadata) if post.metadata else {}
            except Exception:
                pass

            result["client_note"] = {
                "path": rel_path,
                "content": content,
                "frontmatter": sanitize_for_json(fm_data),
            }
            result["sp_folder_id"] = fm_data.get("sp_folder_id", None)

        except Exception as e:
            logger.error("vault_client_context: error reading %s: %s", rel_path, e)
            result["client_note"] = {"path": rel_path, "error": str(e)}
            result["sp_folder_id"] = None

        # Related files: other vault files whose `related:` frontmatter references this client
        try:
            from ..server import frontmatter_index
            client_stem_lower = matched_file.stem.lower()
            rel_path_lower = rel_path.lower()
            related_paths: list[str] = []
            with frontmatter_index._lock:
                for file_rel, fm in frontmatter_index._index.items():
                    if file_rel == rel_path:
                        continue
                    related_val = fm.get("related")
                    if not related_val:
                        continue
                    related_str = str(related_val).lower()
                    if client_stem_lower in related_str or rel_path_lower in related_str:
                        related_paths.append(file_rel)
                        if len(related_paths) >= 10:
                            break
            result["related"] = related_paths
        except Exception as e:
            logger.warning("vault_client_context: related search failed: %s", e)
            result["related"] = []

    # --- Hot.md ---
    if include_hot:
        try:
            content, _ = read_file(HOT_MD_PATH)
            result["hot"] = {"path": HOT_MD_PATH, "content": content}
        except Exception as e:
            result["hot"] = {"path": HOT_MD_PATH, "error": str(e)}

    # --- Lending Analyst instructions ---
    if include_instructions:
        try:
            content, _ = read_file(INSTRUCTIONS_PATH)
            result["instructions"] = {"path": INSTRUCTIONS_PATH, "content": content}
        except Exception as e:
            result["instructions"] = {"path": INSTRUCTIONS_PATH, "error": str(e)}

    return json.dumps(sanitize_for_json(result), cls=SafeJSONEncoder)
