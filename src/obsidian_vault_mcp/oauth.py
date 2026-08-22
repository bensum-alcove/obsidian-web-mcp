"""OAuth 2.1 authorization server with password gate for vault-mcp.

Implements RFC 9728 (protected resource metadata) + RFC 8414 (authorization
server metadata) + authorization code flow with PKCE S256.

Only activates when VAULT_AUTH_PASSWORD is set. When unset, this module is
imported but no routes are mounted (server.py checks before mounting).
"""

import base64
import hashlib
import hmac
import html
import logging
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from . import config

logger = logging.getLogger(__name__)

# --- In-memory stores (auth codes always live here; short 10-min TTL makes
# restart loss harmless) ---

# code -> {client_id, redirect_uri, code_challenge, code_challenge_method, expires_at}
_auth_codes: dict[str, dict] = {}

# token -> expires_at (unix timestamp) — used only when VAULT_TOKEN_DB is unset.
# This is the original pre-persistence store, kept byte-identical for that path.
_tokens: dict[str, float] = {}

# --- SQLite-backed token store (used when VAULT_TOKEN_DB is set) ---
#
# Tokens are stored as SHA-256 hashes only — never plaintext — so a leaked DB
# file grants nothing. Lazy cleanup runs on each write.

_db_lock = threading.Lock()
_db_conn: sqlite3.Connection | None = None


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _get_db() -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        db_path = Path(config.VAULT_TOKEN_DB).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tokens ("
            "token_hash TEXT PRIMARY KEY, kind TEXT NOT NULL, "
            "scope TEXT, expires_at REAL NOT NULL, created_at REAL NOT NULL)"
        )
        conn.commit()
        try:
            db_path.chmod(0o600)
        except OSError:
            pass
        _db_conn = conn
    return _db_conn


def _store_token(token: str, kind: str, ttl_seconds: float, scope: str = "vault") -> None:
    now = time.time()
    if config.VAULT_TOKEN_DB:
        conn = _get_db()
        with _db_lock:
            conn.execute("DELETE FROM tokens WHERE expires_at < ?", (now,))
            conn.execute(
                "INSERT OR REPLACE INTO tokens (token_hash, kind, scope, expires_at, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (_hash_token(token), kind, scope, now + ttl_seconds, now),
            )
            conn.commit()
    else:
        _tokens[token] = now + ttl_seconds


def _consume_refresh_token(token: str) -> bool:
    """Validate and single-use-delete a refresh token (rotation). SQLite-mode only."""
    if not config.VAULT_TOKEN_DB:
        return False
    conn = _get_db()
    token_hash = _hash_token(token)
    with _db_lock:
        row = conn.execute(
            "SELECT expires_at FROM tokens WHERE token_hash = ? AND kind = 'refresh'",
            (token_hash,),
        ).fetchone()
        if row is None or row[0] < time.time():
            return False
        conn.execute("DELETE FROM tokens WHERE token_hash = ?", (token_hash,))
        conn.commit()
    return True


def _cleanup() -> None:
    now = time.time()
    for k in [k for k, v in _auth_codes.items() if v["expires_at"] < now]:
        del _auth_codes[k]
    for k in [k for k, v in _tokens.items() if v < now]:
        del _tokens[k]


def is_valid_oauth_token(token: str) -> bool:
    """Check whether a token is a valid, unexpired OAuth-issued access token."""
    if config.VAULT_TOKEN_DB:
        conn = _get_db()
        with _db_lock:
            row = conn.execute(
                "SELECT expires_at FROM tokens WHERE token_hash = ? AND kind = 'access'",
                (_hash_token(token),),
            ).fetchone()
        return row is not None and row[0] > time.time()
    _cleanup()
    exp = _tokens.get(token)
    return exp is not None and exp > time.time()


def _base_url(request: Request) -> str:
    if config.VAULT_BASE_URL:
        return config.VAULT_BASE_URL.rstrip("/")
    return str(request.base_url).rstrip("/")


# ---------------------------------------------------------------------------
# Discovery endpoints
# ---------------------------------------------------------------------------

async def protected_resource_metadata(request: Request) -> JSONResponse:
    """RFC 9728 — protected resource metadata."""
    base = _base_url(request)
    return JSONResponse({
        "resource": f"{base}/mcp",
        "authorization_servers": [base],
        "scopes_supported": ["vault"],
    })


async def authorization_server_metadata(request: Request) -> JSONResponse:
    """RFC 8414 — authorization server metadata."""
    base = _base_url(request)
    grant_types = ["authorization_code"]
    if config.VAULT_TOKEN_DB:
        grant_types.append("refresh_token")
    return JSONResponse({
        "issuer": base,
        "authorization_endpoint": f"{base}/authorize",
        "token_endpoint": f"{base}/token",
        "response_types_supported": ["code"],
        "grant_types_supported": grant_types,
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": ["vault"],
    })


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------

_LOGIN_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vault — Authorise</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: #1a1a1a;
  color: #e0e0e0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
}}
.card {{
  background: #242424;
  border: 1px solid #333;
  border-radius: 12px;
  padding: 40px;
  width: 100%;
  max-width: 400px;
}}
h1 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 8px; }}
.subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 28px; }}
label {{ display: block; font-size: 0.85rem; color: #aaa; margin-bottom: 6px; }}
input[type="password"] {{
  width: 100%;
  padding: 10px 14px;
  background: #1a1a1a;
  border: 1px solid #444;
  border-radius: 8px;
  color: #e0e0e0;
  font-size: 1rem;
  margin-bottom: 20px;
  outline: none;
}}
input[type="password"]:focus {{ border-color: #666; }}
button {{
  width: 100%;
  padding: 10px;
  background: #3a7bd5;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  cursor: pointer;
}}
button:hover {{ background: #2d6abf; }}
.error {{
  background: #3d1a1a;
  border: 1px solid #7a3030;
  color: #ff8080;
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 0.875rem;
  margin-bottom: 16px;
}}
</style>
</head>
<body>
<div class="card">
  <h1>Vault — Authorise</h1>
  <p class="subtitle">Enter the vault password to grant access.</p>
  {error_block}
  <form method="POST">
    <input type="hidden" name="client_id" value="{client_id}">
    <input type="hidden" name="redirect_uri" value="{redirect_uri}">
    <input type="hidden" name="state" value="{state}">
    <input type="hidden" name="code_challenge" value="{code_challenge}">
    <input type="hidden" name="code_challenge_method" value="{code_challenge_method}">
    <label for="password">Password</label>
    <input type="password" id="password" name="password" autofocus required>
    <button type="submit">Authorise</button>
  </form>
</div>
</body>
</html>"""

_SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-store",
}


def _render_login(
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    code_challenge_method: str,
    error: str = "",
) -> HTMLResponse:
    error_block = f'<div class="error">{error}</div>' if error else ""
    html_content = _LOGIN_HTML.format(
        client_id=html.escape(client_id),
        redirect_uri=html.escape(redirect_uri),
        state=html.escape(state),
        code_challenge=html.escape(code_challenge),
        code_challenge_method=html.escape(code_challenge_method),
        error_block=error_block,
    )
    return HTMLResponse(html_content, headers=_SECURITY_HEADERS)


# ---------------------------------------------------------------------------
# Authorization endpoint — GET (show form) + POST (process submission)
# ---------------------------------------------------------------------------

async def authorize(request: Request) -> Response:
    if request.method == "GET":
        return await _authorize_get(request)
    return await _authorize_post(request)


async def _authorize_get(request: Request) -> Response:
    response_type = request.query_params.get("response_type", "")
    client_id = request.query_params.get("client_id", "")
    redirect_uri = request.query_params.get("redirect_uri", "")
    state = request.query_params.get("state", "")
    code_challenge = request.query_params.get("code_challenge", "")
    code_challenge_method = request.query_params.get("code_challenge_method", "S256")

    if response_type != "code":
        return JSONResponse({"error": "unsupported_response_type"}, status_code=400)
    if not redirect_uri:
        return JSONResponse(
            {"error": "invalid_request", "error_description": "redirect_uri required"},
            status_code=400,
        )
    if code_challenge_method != "S256":
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Only S256 PKCE is supported"},
            status_code=400,
        )
    if config.VAULT_OAUTH_CLIENT_ID and not hmac.compare_digest(client_id, config.VAULT_OAUTH_CLIENT_ID):
        return JSONResponse({"error": "invalid_client"}, status_code=400)

    return _render_login(client_id, redirect_uri, state, code_challenge, code_challenge_method)


async def _authorize_post(request: Request) -> Response:
    try:
        form = await request.form()
    except Exception:
        return JSONResponse({"error": "invalid_request"}, status_code=400)

    client_id = form.get("client_id", "")
    redirect_uri = form.get("redirect_uri", "")
    state = form.get("state", "")
    code_challenge = form.get("code_challenge", "")
    code_challenge_method = form.get("code_challenge_method", "S256")
    password = form.get("password", "")

    if not redirect_uri:
        return JSONResponse({"error": "invalid_request"}, status_code=400)

    if not hmac.compare_digest(password, config.VAULT_AUTH_PASSWORD):
        return _render_login(
            client_id, redirect_uri, state, code_challenge, code_challenge_method,
            "Incorrect password. Please try again.",
        )

    _cleanup()
    code = secrets.token_urlsafe(32)
    _auth_codes[code] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
        "expires_at": time.time() + 600,  # 10 minutes
    }

    logger.info(f"Auth code issued for client_id={client_id!r}")

    params: dict[str, str] = {"code": code}
    if state:
        params["state"] = state
    separator = "&" if "?" in redirect_uri else "?"
    return RedirectResponse(url=f"{redirect_uri}{separator}{urlencode(params)}", status_code=302)


# ---------------------------------------------------------------------------
# Token endpoint
# ---------------------------------------------------------------------------

def _validate_client_secret(request: Request, form, client_id: str) -> JSONResponse | None:
    """Returns an error JSONResponse if client auth fails, else None."""
    client_secret = form.get("client_secret", "")

    # Also accept Basic auth for client_secret
    if not client_secret:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
                _, client_secret = decoded.split(":", 1)
            except Exception:
                pass

    if config.VAULT_OAUTH_CLIENT_SECRET:
        if not client_secret or not hmac.compare_digest(client_secret, config.VAULT_OAUTH_CLIENT_SECRET):
            logger.warning(f"Token request: invalid client_secret for client_id={client_id!r}")
            return JSONResponse(
                {"error": "invalid_client", "error_description": "Client authentication failed"},
                status_code=401,
            )
    return None


def _issue_token_response() -> JSONResponse:
    access_token = secrets.token_hex(32)
    _store_token(access_token, "access", config.VAULT_TOKEN_TTL_SECONDS)

    body = {
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": config.VAULT_TOKEN_TTL_SECONDS,
        "scope": "vault",
    }
    if config.VAULT_TOKEN_DB:
        refresh_token = secrets.token_hex(32)
        _store_token(refresh_token, "refresh", config.VAULT_REFRESH_TTL_SECONDS)
        body["refresh_token"] = refresh_token

    return JSONResponse(body)


async def token_endpoint(request: Request) -> JSONResponse:
    try:
        form = await request.form()
    except Exception:
        return JSONResponse({"error": "invalid_request"}, status_code=400)

    grant_type = form.get("grant_type", "")
    client_id = form.get("client_id", "")

    client_error = _validate_client_secret(request, form, client_id)
    if client_error is not None:
        return client_error

    if grant_type == "refresh_token":
        refresh_token = form.get("refresh_token", "")
        if not refresh_token or not _consume_refresh_token(refresh_token):
            return JSONResponse(
                {"error": "invalid_grant", "error_description": "Invalid or expired refresh token"},
                status_code=400,
            )
        logger.info(f"OAuth token refreshed for client_id={client_id!r}")
        return _issue_token_response()

    if grant_type != "authorization_code":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    code = form.get("code", "")
    redirect_uri = form.get("redirect_uri", "")
    code_verifier = form.get("code_verifier", "")

    _cleanup()

    if code not in _auth_codes:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Invalid or expired code"},
            status_code=400,
        )

    code_data = _auth_codes.pop(code)

    if code_data["expires_at"] < time.time():
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Code expired"},
            status_code=400,
        )

    if redirect_uri and code_data["redirect_uri"] and redirect_uri != code_data["redirect_uri"]:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "redirect_uri mismatch"},
            status_code=400,
        )

    if code_data["client_id"] and client_id and not hmac.compare_digest(code_data["client_id"], client_id):
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "client_id mismatch"},
            status_code=400,
        )

    # Verify PKCE
    if code_data["code_challenge"]:
        if not code_verifier:
            return JSONResponse(
                {"error": "invalid_grant", "error_description": "code_verifier required"},
                status_code=400,
            )
        if code_data["code_challenge_method"] != "S256":
            return JSONResponse(
                {"error": "invalid_grant", "error_description": "Only S256 PKCE supported"},
                status_code=400,
            )
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        if not hmac.compare_digest(computed, code_data["code_challenge"]):
            return JSONResponse(
                {"error": "invalid_grant", "error_description": "PKCE verification failed"},
                status_code=400,
            )

    logger.info(f"OAuth token issued for client_id={client_id!r}")
    return _issue_token_response()


# ---------------------------------------------------------------------------
# Routes — mounted conditionally by server.py when VAULT_AUTH_PASSWORD is set
# ---------------------------------------------------------------------------

oauth_routes = [
    Route("/.well-known/oauth-protected-resource/mcp", protected_resource_metadata, methods=["GET"]),
    Route("/.well-known/oauth-protected-resource", protected_resource_metadata, methods=["GET"]),
    Route("/.well-known/oauth-authorization-server", authorization_server_metadata, methods=["GET"]),
    Route("/authorize", authorize, methods=["GET", "POST"]),
    Route("/token", token_endpoint, methods=["POST"]),
]
