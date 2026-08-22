"""Tests for OAuth token persistence (SQLite store + refresh_token grant)."""

import base64
import hashlib
import sqlite3

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from obsidian_vault_mcp import config, oauth


@pytest.fixture
def oauth_env(monkeypatch, tmp_path):
    """Reset oauth module state and configure a fresh password-gated client."""
    monkeypatch.setattr(config, "VAULT_AUTH_PASSWORD", "correct-horse")
    monkeypatch.setattr(config, "VAULT_OAUTH_CLIENT_ID", "test-client")
    monkeypatch.setattr(config, "VAULT_OAUTH_CLIENT_SECRET", "test-secret")
    monkeypatch.setattr(config, "VAULT_BASE_URL", "")
    monkeypatch.setattr(config, "VAULT_TOKEN_DB", "")
    monkeypatch.setattr(config, "VAULT_TOKEN_TTL_SECONDS", 86400)
    monkeypatch.setattr(config, "VAULT_REFRESH_TTL_SECONDS", 7776000)

    oauth._auth_codes.clear()
    oauth._tokens.clear()
    if oauth._db_conn is not None:
        oauth._db_conn.close()
    oauth._db_conn = None

    app = Starlette(routes=oauth.oauth_routes)
    client = TestClient(app)
    yield client, tmp_path

    if oauth._db_conn is not None:
        oauth._db_conn.close()
    oauth._db_conn = None


def _pkce_pair():
    import secrets

    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _get_auth_code(client, verifier_challenge):
    verifier, challenge = verifier_challenge
    resp = client.post(
        "/authorize",
        data={
            "client_id": "test-client",
            "redirect_uri": "https://example.com/cb",
            "state": "xyz",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "password": "correct-horse",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302
    location = resp.headers["location"]
    code = location.split("code=")[1].split("&")[0]
    return code


def _exchange_code(client, code, verifier):
    return client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": "https://example.com/cb",
            "client_id": "test-client",
            "client_secret": "test-secret",
            "code_verifier": verifier,
        },
    )


def test_in_memory_mode_unchanged(oauth_env):
    """VAULT_TOKEN_DB unset: 24h TTL, no refresh_token, byte-identical to pre-persistence behaviour."""
    client, _ = oauth_env
    pair = _pkce_pair()
    code = _get_auth_code(client, pair)
    resp = _exchange_code(client, code, pair[0])

    assert resp.status_code == 200
    body = resp.json()
    assert body["expires_in"] == 86400
    assert "refresh_token" not in body
    assert oauth.is_valid_oauth_token(body["access_token"]) is True

    meta = client.get("/.well-known/oauth-authorization-server").json()
    assert meta["grant_types_supported"] == ["authorization_code"]


def test_sqlite_persistence_survives_reconnect(oauth_env, monkeypatch):
    """A minted token, and the raw DB contents, must survive a simulated restart."""
    client, tmp_path = oauth_env
    db_path = tmp_path / "tokens.db"
    monkeypatch.setattr(config, "VAULT_TOKEN_DB", str(db_path))
    monkeypatch.setattr(config, "VAULT_TOKEN_TTL_SECONDS", 2592000)

    pair = _pkce_pair()
    code = _get_auth_code(client, pair)
    resp = _exchange_code(client, code, pair[0])
    assert resp.status_code == 200
    body = resp.json()
    access_token = body["access_token"]
    assert body["expires_in"] == 2592000
    assert "refresh_token" in body

    # Only a SHA-256 hash is on disk, never the plaintext token (WAL mode —
    # checkpoint first so the write is flushed to the main db file).
    oauth._get_db().execute("PRAGMA wal_checkpoint(TRUNCATE)")
    raw = db_path.read_bytes()
    assert access_token.encode("ascii") not in raw
    assert hashlib.sha256(access_token.encode()).hexdigest().encode("ascii") in raw

    assert oauth.is_valid_oauth_token(access_token) is True

    # Simulate a service restart: drop the in-process connection.
    oauth._db_conn.close()
    oauth._db_conn = None

    assert oauth.is_valid_oauth_token(access_token) is True


def test_expiry_honoured_both_modes(oauth_env, monkeypatch, tmp_path):
    client, _ = oauth_env

    oauth._store_token("expired-inmem", "access", -10)
    assert oauth.is_valid_oauth_token("expired-inmem") is False

    monkeypatch.setattr(config, "VAULT_TOKEN_DB", str(tmp_path / "tokens.db"))
    oauth._store_token("expired-sqlite", "access", -10)
    assert oauth.is_valid_oauth_token("expired-sqlite") is False


def test_refresh_rotation(oauth_env, monkeypatch, tmp_path):
    """Refresh grant issues a new pair and invalidates the old refresh token."""
    client, _ = oauth_env
    monkeypatch.setattr(config, "VAULT_TOKEN_DB", str(tmp_path / "tokens.db"))

    pair = _pkce_pair()
    code = _get_auth_code(client, pair)
    first = _exchange_code(client, code, pair[0]).json()
    old_refresh = first["refresh_token"]

    refreshed = client.post(
        "/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": old_refresh,
            "client_id": "test-client",
            "client_secret": "test-secret",
        },
    )
    assert refreshed.status_code == 200
    new_body = refreshed.json()
    assert new_body["access_token"] != first["access_token"]
    assert new_body["refresh_token"] != old_refresh
    assert oauth.is_valid_oauth_token(new_body["access_token"]) is True

    # Old refresh token is single-use — rejected on reuse.
    reuse = client.post(
        "/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": old_refresh,
            "client_id": "test-client",
            "client_secret": "test-secret",
        },
    )
    assert reuse.status_code == 400
    assert reuse.json()["error"] == "invalid_grant"


def test_refresh_grant_rejected_when_token_db_unset(oauth_env):
    """Without VAULT_TOKEN_DB there is no refresh store — grant must fail cleanly."""
    client, _ = oauth_env
    resp = client.post(
        "/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": "anything",
            "client_id": "test-client",
            "client_secret": "test-secret",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_grant"


def test_garbage_and_missing_token_invalid(oauth_env, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "VAULT_TOKEN_DB", str(tmp_path / "tokens.db"))
    assert oauth.is_valid_oauth_token("not-a-real-token") is False


def test_alcove_path_untouched_when_no_password(monkeypatch):
    """When VAULT_AUTH_PASSWORD is unset, auth.py must never import/touch oauth at all."""
    from obsidian_vault_mcp import auth as auth_module

    monkeypatch.setattr(auth_module, "VAULT_AUTH_PASSWORD", "")
    monkeypatch.setattr(auth_module, "VAULT_MCP_TOKEN", "legacy-token")

    async def handler(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/mcp", handler)])
    app.add_middleware(auth_module.BearerAuthMiddleware)
    client = TestClient(app)

    resp = client.get("/mcp", headers={"Authorization": "Bearer legacy-token"})
    assert resp.status_code == 200

    resp_bad = client.get("/mcp", headers={"Authorization": "Bearer wrong"})
    assert resp_bad.status_code == 401
    assert "WWW-Authenticate" not in resp_bad.headers
