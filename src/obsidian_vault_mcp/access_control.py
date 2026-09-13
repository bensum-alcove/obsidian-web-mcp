"""Central, explicit read/mutation classification for every registered MCP tool.

This is the single source of truth for which tools are safe to expose on a
read-only deployment (VAULT_ACCESS_MODE=read_only, e.g. cb-brain-marketing-ro).
A tool name with no entry here is a build-time error, not a silent gap: every
`@tool_gate(...)` call site in server.py runs through `classify()`, which
raises if the name is unlisted -- so an unclassified tool (including one added
in the future) fails server startup in *every* access mode rather than
silently appearing in a read-only deployment. This is deliberately not a
name-prefix/substring heuristic (e.g. "starts with vault_" or "doesn't say
write") -- classification is a fixed, reviewed table.
"""

from enum import Enum


class AccessClass(str, Enum):
    READ = "read"
    MUTATION = "mutation"


VALID_ACCESS_MODES = ("full", "read_only")

# Every tool registered via @tool_gate(...) in server.py must appear here.
# "read": never writes vault content, never mutates BO schedule/spec state.
# "mutation": writes vault content and/or BO schedule/spec state (directly,
# or indirectly via a nested helper -- see the classification evidence in
# this build's output doc for the file:line trace behind each entry).
TOOL_ACCESS_CLASS: dict[str, AccessClass] = {
    # --- Pure reads ---
    "vault_read": AccessClass.READ,
    "vault_batch_read": AccessClass.READ,
    "vault_search": AccessClass.READ,
    "vault_search_frontmatter": AccessClass.READ,
    "vault_list": AccessClass.READ,
    "vault_read_section": AccessClass.READ,
    "vault_recent_changes": AccessClass.READ,
    "vault_stats": AccessClass.READ,
    "vault_session_start": AccessClass.READ,
    "vault_client_context": AccessClass.READ,
    "vault_entity": AccessClass.READ,
    "vault_query": AccessClass.READ,
    "vault_answer_context": AccessClass.READ,
    "bo_validate_build_graph": AccessClass.READ,
    "vault_semantic_search": AccessClass.READ,
    "vault_read_smart": AccessClass.READ,
    # --- Mutations ---
    "vault_write": AccessClass.MUTATION,
    "vault_batch_frontmatter_update": AccessClass.MUTATION,
    "vault_move": AccessClass.MUTATION,
    "vault_delete": AccessClass.MUTATION,
    "vault_patch_section": AccessClass.MUTATION,
    "vault_append": AccessClass.MUTATION,
    "vault_batch_write": AccessClass.MUTATION,
    "vault_str_replace": AccessClass.MUTATION,
    "vault_batch_delete": AccessClass.MUTATION,
    "vault_batch_str_replace": AccessClass.MUTATION,
    "bo_create_build": AccessClass.MUTATION,
    "bo_create_chain": AccessClass.MUTATION,
    "bo_activate_existing_spec": AccessClass.MUTATION,
}


def classify(name: str) -> AccessClass:
    """Look up a tool's access class. Raises if unclassified -- fail closed."""
    try:
        return TOOL_ACCESS_CLASS[name]
    except KeyError as exc:
        raise RuntimeError(
            f"Tool {name!r} has no entry in access_control.TOOL_ACCESS_CLASS. "
            "Every MCP tool must be explicitly classified read/mutation before "
            "registration -- add it to the table before deploying."
        ) from exc


def should_register(name: str, access_mode: str) -> bool:
    """Whether `name` should be registered on the live MCP tool surface for
    `access_mode`. Fail closed: an unrecognized access_mode raises rather than
    silently defaulting to the more permissive "full" behavior.
    """
    if access_mode not in VALID_ACCESS_MODES:
        raise RuntimeError(
            f"Unknown VAULT_ACCESS_MODE={access_mode!r}; expected one of {VALID_ACCESS_MODES}."
        )
    access_class = classify(name)
    if access_mode == "full":
        return True
    return access_class is AccessClass.READ
