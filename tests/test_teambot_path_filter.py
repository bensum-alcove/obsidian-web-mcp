"""Tests for teambot path filtering and auth enforcement.

All paths use the real vault layout: files live under BS 2nd Brain/Alcove/...
relative to the vault root.
"""

import json
from pathlib import Path

import pytest


# --- Unit tests for validate_teambot_path ---

def test_allowed_clients_path():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    validate_teambot_path("BS 2nd Brain/Alcove/Clients/SomeClient/notes.md")


def test_allowed_clients_directory():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    validate_teambot_path("BS 2nd Brain/Alcove/Clients")


def test_allowed_todo_path():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    validate_teambot_path("BS 2nd Brain/Alcove/Operations/Todo/2026-07-06.md")


def test_allowed_triage_path():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    validate_teambot_path("BS 2nd Brain/Alcove/Triage/loan-book.md")


def test_allowed_triage_directory():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    validate_teambot_path("BS 2nd Brain/Alcove/Triage")


def test_blocked_root_empty_path():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("")


def test_blocked_hot_md():
    """BS 2nd Brain/Alcove/Skills/hot.md is the known out-of-scope leak path."""
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("BS 2nd Brain/Alcove/Skills/hot.md")


def test_blocked_infrastructure():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("BS 2nd Brain/Alcove/Infrastructure/anything.md")


def test_blocked_mah_mahs_estate():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("BS 2nd Brain/MahMahs Estate/something.md")


def test_blocked_personal():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("Personal/Build Orchestrator/specs/secret.md")


def test_blocked_root_relative_clients():
    """Root-relative paths (without BS 2nd Brain/) must be denied."""
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("Alcove/Clients/foo.md")


def test_blocked_root_relative_todo():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("Alcove/Operations/Todo/2026-07-07.md")


def test_blocked_traversal_from_clients():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("BS 2nd Brain/Alcove/Clients/../../Personal/secret.md")


def test_blocked_traversal_absolute_style():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("../../../etc/passwd")


def test_blocked_sibling_prefix_attack():
    """'BS 2nd Brain/Alcove/ClientsEvil' must not match 'BS 2nd Brain/Alcove/Clients'."""
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("BS 2nd Brain/Alcove/ClientsEvil/foo.md")


def test_blocked_operations_not_todo():
    """'BS 2nd Brain/Alcove/Operations/Other' must not match 'BS 2nd Brain/Alcove/Operations/Todo'."""
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("BS 2nd Brain/Alcove/Operations/Other/file.md")


def test_blocked_triage_sibling():
    """'BS 2nd Brain/Alcove/TriageOther' must not match 'BS 2nd Brain/Alcove/Triage'."""
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("BS 2nd Brain/Alcove/TriageOther/file.md")


# --- Integration tests: auth enforcement + scope via TestClient ---

@pytest.fixture
def teambot_vault(tmp_path, monkeypatch):
    """Temporary vault with real BS 2nd Brain layout — in-scope and out-of-scope files."""
    # In-scope: Clients
    clients_dir = tmp_path / "BS 2nd Brain" / "Alcove" / "Clients" / "TestCorp"
    clients_dir.mkdir(parents=True)
    (clients_dir / "notes.md").write_text("# TestCorp\nClient note content.")

    # In-scope: Operations/Todo
    todo_dir = tmp_path / "BS 2nd Brain" / "Alcove" / "Operations" / "Todo"
    todo_dir.mkdir(parents=True)
    (todo_dir / "2026-07-06.md").write_text("# Tasks\n- [ ] do the thing")

    # In-scope: Triage
    triage_dir = tmp_path / "BS 2nd Brain" / "Alcove" / "Triage"
    triage_dir.mkdir(parents=True)
    (triage_dir / "loan-book.md").write_text("# Loan Book\n| Client |\n|--------|\n| Test |")

    # Out-of-scope: Skills (the known hot.md path)
    skills_dir = tmp_path / "BS 2nd Brain" / "Alcove" / "Skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "hot.md").write_text("# Hot\nDo not expose.")

    # Out-of-scope: MahMahs Estate
    estate_dir = tmp_path / "BS 2nd Brain" / "MahMahs Estate"
    estate_dir.mkdir(parents=True)
    (estate_dir / "private.md").write_text("# Private\nDo not expose.")

    monkeypatch.setenv("VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("VAULT_MCP_TOKEN", "main-token")
    monkeypatch.setenv("TEAMBOT_MCP_TOKEN", "teambot-token")

    import obsidian_vault_mcp.config as cfg
    cfg.VAULT_PATH = tmp_path
    cfg.VAULT_MCP_TOKEN = "main-token"
    cfg.TEAMBOT_MCP_TOKEN = "teambot-token"

    return tmp_path


def test_teambot_no_token_returns_401(teambot_vault):
    from starlette.testclient import TestClient
    import obsidian_vault_mcp.teambot as tb_module
    import importlib
    importlib.reload(tb_module)

    teambot_app = tb_module.build_teambot_app()
    with TestClient(teambot_app, base_url="http://localhost", raise_server_exceptions=False) as client:
        resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                          "params": {"protocolVersion": "2024-11-05",
                                                     "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}})
        assert resp.status_code == 401


def test_teambot_wrong_token_returns_401(teambot_vault):
    from starlette.testclient import TestClient
    import obsidian_vault_mcp.teambot as tb_module
    import importlib
    importlib.reload(tb_module)

    teambot_app = tb_module.build_teambot_app()
    with TestClient(teambot_app, base_url="http://localhost", raise_server_exceptions=False) as client:
        resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                          "params": {"protocolVersion": "2024-11-05",
                                                     "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}},
                           headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code == 401


_MCP_HEADERS = {
    "Authorization": "Bearer teambot-token",
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def test_teambot_in_scope_read_clients_succeeds(teambot_vault):
    from starlette.testclient import TestClient
    import obsidian_vault_mcp.teambot as tb_module
    import importlib
    importlib.reload(tb_module)

    teambot_app = tb_module.build_teambot_app()
    with TestClient(teambot_app, base_url="http://localhost", raise_server_exceptions=False) as client:
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "vault_read",
                           "arguments": {"path": "BS 2nd Brain/Alcove/Clients/TestCorp/notes.md"}},
            },
            headers=_MCP_HEADERS,
        )
        assert resp.status_code == 200
        body = json.dumps(resp.json())
        assert "outside teambot scope" not in body


def test_teambot_in_scope_read_triage_succeeds(teambot_vault):
    from starlette.testclient import TestClient
    import obsidian_vault_mcp.teambot as tb_module
    import importlib
    importlib.reload(tb_module)

    teambot_app = tb_module.build_teambot_app()
    with TestClient(teambot_app, base_url="http://localhost", raise_server_exceptions=False) as client:
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "vault_read",
                           "arguments": {"path": "BS 2nd Brain/Alcove/Triage/loan-book.md"}},
            },
            headers=_MCP_HEADERS,
        )
        assert resp.status_code == 200
        body = json.dumps(resp.json())
        assert "outside teambot scope" not in body


def test_teambot_out_of_scope_hot_md_denied(teambot_vault):
    """The known leak path — BS 2nd Brain/Alcove/Skills/hot.md — must be denied."""
    from starlette.testclient import TestClient
    import obsidian_vault_mcp.teambot as tb_module
    import importlib
    importlib.reload(tb_module)

    teambot_app = tb_module.build_teambot_app()
    with TestClient(teambot_app, base_url="http://localhost", raise_server_exceptions=False) as client:
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "vault_read",
                           "arguments": {"path": "BS 2nd Brain/Alcove/Skills/hot.md"}},
            },
            headers=_MCP_HEADERS,
        )
        assert resp.status_code == 200
        body = json.dumps(resp.json())
        assert "outside teambot scope" in body


def test_teambot_out_of_scope_estate_denied(teambot_vault):
    from starlette.testclient import TestClient
    import obsidian_vault_mcp.teambot as tb_module
    import importlib
    importlib.reload(tb_module)

    teambot_app = tb_module.build_teambot_app()
    with TestClient(teambot_app, base_url="http://localhost", raise_server_exceptions=False) as client:
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "vault_read",
                           "arguments": {"path": "BS 2nd Brain/MahMahs Estate/private.md"}},
            },
            headers=_MCP_HEADERS,
        )
        assert resp.status_code == 200
        body = json.dumps(resp.json())
        assert "outside teambot scope" in body


def test_teambot_path_traversal_denied(teambot_vault):
    from starlette.testclient import TestClient
    import obsidian_vault_mcp.teambot as tb_module
    import importlib
    importlib.reload(tb_module)

    teambot_app = tb_module.build_teambot_app()
    with TestClient(teambot_app, base_url="http://localhost", raise_server_exceptions=False) as client:
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": {"name": "vault_read",
                           "arguments": {"path": "BS 2nd Brain/Alcove/Clients/../Skills/hot.md"}},
            },
            headers=_MCP_HEADERS,
        )
        assert resp.status_code == 200
        body = json.dumps(resp.json())
        assert "outside teambot scope" in body


def test_teambot_search_in_scope_triage(teambot_vault):
    from starlette.testclient import TestClient
    import obsidian_vault_mcp.teambot as tb_module
    import importlib
    importlib.reload(tb_module)

    teambot_app = tb_module.build_teambot_app()
    with TestClient(teambot_app, base_url="http://localhost", raise_server_exceptions=False) as client:
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0", "id": 5, "method": "tools/call",
                "params": {"name": "vault_search",
                           "arguments": {"query": "Loan", "path_prefix": "BS 2nd Brain/Alcove/Triage"}},
            },
            headers=_MCP_HEADERS,
        )
        assert resp.status_code == 200
        body = json.dumps(resp.json())
        assert "outside teambot scope" not in body


def test_teambot_search_out_of_scope_denied(teambot_vault):
    from starlette.testclient import TestClient
    import obsidian_vault_mcp.teambot as tb_module
    import importlib
    importlib.reload(tb_module)

    teambot_app = tb_module.build_teambot_app()
    with TestClient(teambot_app, base_url="http://localhost", raise_server_exceptions=False) as client:
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0", "id": 6, "method": "tools/call",
                "params": {"name": "vault_search",
                           "arguments": {"query": "hot", "path_prefix": "BS 2nd Brain/Alcove/Skills"}},
            },
            headers=_MCP_HEADERS,
        )
        assert resp.status_code == 200
        body = json.dumps(resp.json())
        assert "outside teambot scope" in body
