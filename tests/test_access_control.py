"""Tests for the read/mutation tool-access classification and read_only mode.

Server-registration tests run in a subprocess (like the mode-switch already
requires in production -- VAULT_ACCESS_MODE is read once at process import
time) so each mode gets a clean interpreter rather than relying on module
reload/monkeypatch order.
"""

import json
import subprocess
import sys

import pytest

from obsidian_vault_mcp.access_control import (
    TOOL_ACCESS_CLASS,
    AccessClass,
    classify,
    should_register,
)

KNOWN_MUTATION_TOOLS = {
    "vault_write",
    "vault_batch_frontmatter_update",
    "vault_move",
    "vault_delete",
    "vault_patch_section",
    "vault_append",
    "vault_batch_write",
    "vault_str_replace",
    "vault_batch_delete",
    "vault_batch_str_replace",
    "bo_create_build",
    "bo_create_chain",
    "bo_activate_existing_spec",
}

KNOWN_READ_TOOLS = {
    "vault_read",
    "vault_batch_read",
    "vault_search",
    "vault_search_frontmatter",
    "vault_list",
    "vault_read_section",
    "vault_recent_changes",
    "vault_stats",
    "vault_session_start",
    "vault_client_context",
    "vault_entity",
    "vault_query",
    "vault_answer_context",
    "bo_validate_build_graph",
    "vault_semantic_search",
    "vault_read_smart",
}


def test_classification_table_exhaustive_and_partitioned():
    """Every tool is classified exactly once as read xor mutation."""
    assert set(TOOL_ACCESS_CLASS) == KNOWN_READ_TOOLS | KNOWN_MUTATION_TOOLS
    assert KNOWN_READ_TOOLS.isdisjoint(KNOWN_MUTATION_TOOLS)
    for name in KNOWN_READ_TOOLS:
        assert TOOL_ACCESS_CLASS[name] is AccessClass.READ
    for name in KNOWN_MUTATION_TOOLS:
        assert TOOL_ACCESS_CLASS[name] is AccessClass.MUTATION


def test_classify_raises_for_unknown_tool():
    with pytest.raises(RuntimeError, match="no entry in access_control"):
        classify("vault_definitely_not_a_real_tool")


def test_should_register_full_mode_allows_everything():
    for name in TOOL_ACCESS_CLASS:
        assert should_register(name, "full") is True


def test_should_register_read_only_mode_matches_classification_exactly():
    read_only_allowed = {n for n in TOOL_ACCESS_CLASS if should_register(n, "read_only")}
    assert read_only_allowed == KNOWN_READ_TOOLS


def test_should_register_read_only_mode_excludes_every_mutation_tool():
    for name in KNOWN_MUTATION_TOOLS:
        assert should_register(name, "read_only") is False


def test_should_register_unknown_access_mode_fails_closed():
    with pytest.raises(RuntimeError, match="Unknown VAULT_ACCESS_MODE"):
        should_register("vault_read", "read_write_oops")


def test_should_register_unclassified_tool_fails_closed_in_every_mode():
    for mode in ("full", "read_only"):
        with pytest.raises(RuntimeError, match="no entry in access_control"):
            should_register("some_future_tool_nobody_classified", mode)


def _run_server_subprocess(vault_dir, extra_env, code):
    env = {
        "VAULT_PATH": str(vault_dir),
        "VAULT_MCP_TOKEN": "test-token",
        **extra_env,
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    return result.stdout


def test_server_full_mode_registers_all_classified_tools(tmp_path):
    out = _run_server_subprocess(
        tmp_path,
        {},
        "import json, obsidian_vault_mcp.server as s; "
        "print(json.dumps(sorted(s.mcp._tool_manager._tools.keys())))",
    )
    names = set(json.loads(out.strip().splitlines()[-1]))
    assert names == set(TOOL_ACCESS_CLASS)


def test_server_read_only_mode_registers_exactly_read_tools(tmp_path):
    out = _run_server_subprocess(
        tmp_path,
        {"VAULT_ACCESS_MODE": "read_only"},
        "import json, obsidian_vault_mcp.server as s; "
        "print(json.dumps(sorted(s.mcp._tool_manager._tools.keys())))",
    )
    names = set(json.loads(out.strip().splitlines()[-1]))
    assert names == KNOWN_READ_TOOLS
    assert names.isdisjoint(KNOWN_MUTATION_TOOLS)


def test_server_read_only_mode_mutation_tool_is_unknown_not_forbidden(tmp_path):
    """Direct dispatch by a known mutation tool name must look like the tool
    doesn't exist (KeyError / get() -> None), not a runtime permission denial --
    it was never registered, so there's no permission check to deny it."""
    out = _run_server_subprocess(
        tmp_path,
        {"VAULT_ACCESS_MODE": "read_only"},
        "import obsidian_vault_mcp.server as s; "
        "print('MISSING' if s.mcp._tool_manager._tools.get('vault_write') is None else 'PRESENT')",
    )
    assert out.strip().splitlines()[-1] == "MISSING"


def test_server_invalid_access_mode_fails_closed_at_import(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import obsidian_vault_mcp.server"],
        env={"VAULT_PATH": str(tmp_path), "VAULT_MCP_TOKEN": "t", "VAULT_ACCESS_MODE": "bogus"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "Unknown VAULT_ACCESS_MODE" in result.stderr


def test_build_app_read_only_mode_skips_teambot(tmp_path):
    out = _run_server_subprocess(
        tmp_path,
        {"VAULT_ACCESS_MODE": "read_only"},
        "import obsidian_vault_mcp.server as s; "
        "app = s.build_app('read_only'); "
        "print(type(app).__name__)",
    )
    # Plain Starlette app, NOT TeamBotSiblingDispatcher -- no teambot write-tool
    # object exists in this process at all when access_mode='read_only'.
    assert out.strip().splitlines()[-1] != "TeamBotSiblingDispatcher"


def test_build_app_full_mode_includes_teambot(tmp_path):
    out = _run_server_subprocess(
        tmp_path,
        {},
        "import obsidian_vault_mcp.server as s; "
        "app = s.build_app('full'); "
        "print(type(app).__name__)",
    )
    assert out.strip().splitlines()[-1] == "TeamBotSiblingDispatcher"


def test_semantic_index_path_redirected_by_service_state_dir(tmp_path, monkeypatch):
    # Direct attribute monkeypatch (matches conftest.py's vault_dir fixture
    # convention) rather than importlib.reload() -- reloading config.py mutates
    # the single shared module object for the rest of the test session and
    # leaks into unrelated tests that read config.VAULT_PATH afterward.
    from pathlib import Path
    import obsidian_vault_mcp.config as config
    from obsidian_vault_mcp.tools import semantic_search as ss

    state_dir = tmp_path / "state"
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    monkeypatch.setattr(config, "VAULT_PATH", Path(vault_dir))
    monkeypatch.setattr(config, "VAULT_SERVICE_STATE_DIR", str(state_dir))

    path = ss._get_index_path()
    assert str(path).startswith(str(state_dir))
    assert not str(path).startswith(str(vault_dir))
