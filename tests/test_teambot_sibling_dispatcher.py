"""Regression tests for TeamBotSiblingDispatcher's dual-app ASGI lifespan.

Root cause of the 2026-08-24 outage: the dispatcher's `_lifespan()` skipped the
initial `receive()` that consumes the ASGI `lifespan.startup` message. The next
`receive()` — intended to block until real shutdown — was instead fed that
leftover `lifespan.startup` message, was read as a shutdown signal, and tore
down both sub-apps' StreamableHTTPSessionManager task groups immediately after
starting them. Every request afterward (on both the main route and the
teambot route) failed with "RuntimeError: Task group is not initialized."

These tests build the dispatcher exactly as `main()` does (a real main-style
FastMCP app plus the real `build_teambot_app()`) and drive it through
`TestClient`, which performs a full ASGI lifespan startup/shutdown cycle on
`__enter__`/`__exit__` — the same protocol uvicorn drives in production.
"""

import pytest
from starlette.testclient import TestClient


INIT_PAYLOAD = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "regression-test", "version": "1.0"},
    },
}

MCP_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


@pytest.fixture
def teambot_env(tmp_path, monkeypatch):
    """Minimal vault + tokens, mirroring teambot_vault in test_teambot_path_filter.py."""
    todo_dir = tmp_path / "BS 2nd Brain" / "Alcove" / "Operations" / "Todo"
    todo_dir.mkdir(parents=True)
    (todo_dir / "tasks.md").write_text("# Tasks\n- [ ] do the thing")

    monkeypatch.setenv("VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("VAULT_MCP_TOKEN", "main-token")
    monkeypatch.setenv("TEAMBOT_MCP_TOKEN", "teambot-token")

    import obsidian_vault_mcp.config as cfg
    cfg.VAULT_PATH = tmp_path
    cfg.VAULT_MCP_TOKEN = "main-token"
    cfg.TEAMBOT_MCP_TOKEN = "teambot-token"
    return tmp_path


def _build_dispatcher():
    """Build a fresh main app + fresh teambot app + a fresh dispatcher, exactly
    as server.main() does. Each call yields brand-new FastMCP/session-manager
    instances, since a StreamableHTTPSessionManager can only run() once —
    calling this twice models what actually happens across a process restart.
    """
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
    from obsidian_vault_mcp.server import TeamBotSiblingDispatcher
    from obsidian_vault_mcp.teambot import build_teambot_app

    main_mcp = FastMCP(
        "test_main",
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost", "localhost:*", "[::1]:*"],
        ),
    )

    @main_mcp.tool(name="ping")
    def ping() -> str:
        return "pong"

    main_app = main_mcp.streamable_http_app()
    teambot_app = build_teambot_app()
    return TeamBotSiblingDispatcher(main_app, teambot_app)


def test_main_route_responds_after_startup(teambot_env):
    dispatcher = _build_dispatcher()
    with TestClient(dispatcher, base_url="http://localhost", raise_server_exceptions=False) as client:
        resp = client.post("/mcp", json=INIT_PAYLOAD, headers=MCP_HEADERS)
        assert resp.status_code == 200, resp.text


def test_teambot_route_responds_after_startup(teambot_env):
    dispatcher = _build_dispatcher()
    with TestClient(dispatcher, base_url="http://localhost", raise_server_exceptions=False) as client:
        resp = client.post(
            "/mcp/teambot",
            json=INIT_PAYLOAD,
            headers={**MCP_HEADERS, "Authorization": "Bearer teambot-token"},
        )
        assert resp.status_code == 200, resp.text


def test_both_routes_respond_in_the_same_process(teambot_env):
    """The bug tore down BOTH session managers together — assert neither route
    regresses the other within a single running process."""
    dispatcher = _build_dispatcher()
    with TestClient(dispatcher, base_url="http://localhost", raise_server_exceptions=False) as client:
        main_resp = client.post("/mcp", json=INIT_PAYLOAD, headers=MCP_HEADERS)
        teambot_resp = client.post(
            "/mcp/teambot",
            json=INIT_PAYLOAD,
            headers={**MCP_HEADERS, "Authorization": "Bearer teambot-token"},
        )
        assert main_resp.status_code == 200, main_resp.text
        assert teambot_resp.status_code == 200, teambot_resp.text


def test_teambot_route_survives_a_process_restart(teambot_env):
    """A fresh dispatcher instance (as a real supervisor restart produces) must
    independently reach a working state — this is the exact scenario that
    broke in production: healthy before a restart, broken after."""
    for _ in range(2):
        dispatcher = _build_dispatcher()
        with TestClient(dispatcher, base_url="http://localhost", raise_server_exceptions=False) as client:
            resp = client.post(
                "/mcp/teambot",
                json=INIT_PAYLOAD,
                headers={**MCP_HEADERS, "Authorization": "Bearer teambot-token"},
            )
            assert resp.status_code == 200, resp.text


def test_teambot_route_rejects_invalid_token_through_the_dispatcher(teambot_env):
    """Confirms the sibling dispatcher's path-rewrite doesn't bypass the
    teambot sub-app's own TeamBotBearerAuthMiddleware."""
    dispatcher = _build_dispatcher()
    with TestClient(dispatcher, base_url="http://localhost", raise_server_exceptions=False) as client:
        resp = client.post(
            "/mcp/teambot",
            json=INIT_PAYLOAD,
            headers={**MCP_HEADERS, "Authorization": "Bearer totally-invalid-token"},
        )
        assert resp.status_code == 401, resp.text
