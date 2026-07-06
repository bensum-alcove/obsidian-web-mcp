"""Tests for teambot path filtering and auth enforcement."""

import json
from pathlib import Path

import pytest


# --- Unit tests for validate_teambot_path ---

def test_allowed_clients_path():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    validate_teambot_path("Alcove/Clients/SomeClient/notes.md")


def test_allowed_clients_directory():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    validate_teambot_path("Alcove/Clients")


def test_allowed_todo_path():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    validate_teambot_path("Alcove/Operations/Todo/2026-07-06.md")


def test_blocked_root_empty_path():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("")


def test_blocked_out_of_scope():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("Personal/Build Orchestrator/specs/secret.md")


def test_blocked_alcove_infrastructure():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("Alcove/Infrastructure/anything.md")


def test_blocked_traversal_from_clients():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("Alcove/Clients/../../Personal/secret.md")


def test_blocked_traversal_absolute_style():
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("../../../etc/passwd")


def test_blocked_sibling_prefix_attack():
    """'Alcove/ClientsEvil' must not match 'Alcove/Clients'."""
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("Alcove/ClientsEvil/foo.md")


def test_blocked_operations_not_todo():
    """'Alcove/Operations/Other' must not match 'Alcove/Operations/Todo'."""
    from obsidian_vault_mcp.teambot import validate_teambot_path
    with pytest.raises(ValueError):
        validate_teambot_path("Alcove/Operations/Other/file.md")


# --- Integration tests: auth enforcement + scope via TestClient ---

@pytest.fixture
def teambot_vault(tmp_path, monkeypatch):
    """Temporary vault with in-scope and out-of-scope files."""
    clients_dir = tmp_path / "Alcove" / "Clients" / "TestCorp"
    clients_dir.mkdir(parents=True)
    (clients_dir / "notes.md").write_text("# TestCorp\nClient note content.")

    todo_dir = tmp_path / "Alcove" / "Operations" / "Todo"
    todo_dir.mkdir(parents=True)
    (todo_dir / "2026-07-06.md").write_text("# Tasks\n- [ ] do the thing")

    personal_dir = tmp_path / "Personal"
    personal_dir.mkdir()
    (personal_dir / "secret.md").write_text("# Secret\nDo not expose.")

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


def test_teambot_in_scope_read_succeeds(teambot_vault):
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
                           "arguments": {"path": "Alcove/Clients/TestCorp/notes.md"}},
            },
            headers=_MCP_HEADERS,
        )
        assert resp.status_code == 200
        body = json.dumps(resp.json())
        assert "outside teambot scope" not in body


def test_teambot_out_of_scope_read_denied(teambot_vault):
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
                           "arguments": {"path": "Personal/secret.md"}},
            },
            headers=_MCP_HEADERS,
        )
        assert resp.status_code == 200  # MCP errors are in-band as 200
        body = json.dumps(resp.json())
        assert "outside teambot scope" in body


def test_teambot_search_requires_path_prefix(teambot_vault):
    from starlette.testclient import TestClient
    import obsidian_vault_mcp.teambot as tb_module
    import importlib
    importlib.reload(tb_module)

    teambot_app = tb_module.build_teambot_app()
    with TestClient(teambot_app, base_url="http://localhost", raise_server_exceptions=False) as client:
        # path_prefix pointing outside scope should be denied
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "vault_search",
                           "arguments": {"query": "secret", "path_prefix": "Personal"}},
            },
            headers=_MCP_HEADERS,
        )
        assert resp.status_code == 200
        body = json.dumps(resp.json())
        assert "outside teambot scope" in body
