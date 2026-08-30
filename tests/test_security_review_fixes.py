"""Regression tests for findings from the cb-brain-marketing-readonly-mcp-v1
Codex xhigh security review (2026-08-30). Each test is named after the
finding it locks in so a future regression is traceable back to the review.
"""

import json
import subprocess
import sys

import pytest

from obsidian_vault_mcp.tools.search import _search_ripgrep
from obsidian_vault_mcp.tools.read import vault_read as _vault_read
from obsidian_vault_mcp.tools.search import vault_search


# --- BLOCKER-1: ripgrep option injection via a caller-controlled query ---

def test_blocker1_query_never_placed_where_rg_parses_it_as_an_option(monkeypatch):
    """A query beginning with `-` must be pinned as a pattern (`-e query --`),
    never left where ripgrep's own argument parser could read it as a flag."""
    captured = {}

    class FakeResult:
        stdout = ""
        returncode = 0

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeResult()

    monkeypatch.setattr(subprocess, "run", fake_run)

    from pathlib import Path
    _search_ripgrep("--pre=/bin/true", Path("/tmp/some-vault"), "*.md", 20, 2)

    cmd = captured["cmd"]
    e_idx = cmd.index("-e")
    assert cmd[e_idx + 1] == "--pre=/bin/true"
    dashdash_idx = cmd.index("--", e_idx)
    assert dashdash_idx == e_idx + 2, "-- must immediately follow the pinned query"
    assert "--no-config" in cmd


def test_blocker1_end_to_end_dash_prefixed_query_returns_normally(vault_dir):
    """A `-`-prefixed query against a real vault must behave like an ordinary
    (non-matching) search, not error out or do anything unexpected."""
    result = json.loads(vault_search("--pre=/bin/true"))
    assert "error" not in result
    assert result["results"] == []


# --- HIGH-4: unauthenticated fallback on app-construction failure ---

def test_high4_main_has_no_unauthenticated_fallback():
    """main() must not contain a bare mcp.run() fallback call -- auth
    construction failures must fail closed (process exit), never silently
    serve every tool with no auth middleware. AST-based (not substring) so
    this doesn't false-positive on the explanatory comment referencing the
    removed pattern by name."""
    import ast
    import inspect
    from obsidian_vault_mcp import server

    tree = ast.parse(inspect.getsource(server.main))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "run"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "mcp"
    ]
    assert calls == [], "main() must not call bare mcp.run()"
    # No bare except-Exception around the server startup either.
    assert not any(isinstance(n, ast.ExceptHandler) for n in ast.walk(tree))


# Subprocess isolation for anything that imports/mutates obsidian_vault_mcp.server
# itself: that module runs a lot of side-effecting top-level registration code,
# and reload()-ing it in-process leaks shared global state (the `mcp` FastMCP
# instance, its tool registry) into whichever test runs next in the same
# session -- a real regression a prior test in this same build already hit.
# A fresh interpreter per case matches how each access mode actually runs in
# production (one process, one VAULT_ACCESS_MODE, set once at start).
def _run_server_subprocess(vault_dir, extra_env, code):
    env = {"VAULT_PATH": str(vault_dir), "VAULT_MCP_TOKEN": "test-token", **extra_env}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    return result.stdout


# --- MEDIUM-5: runtime registry audit (not just decorator convention) ---

def test_medium5_audit_catches_a_tool_registered_outside_tool_gate(tmp_path):
    code = (
        "import obsidian_vault_mcp.server as s\n"
        "@s.mcp.tool(name='vault_write', description='bypassed registration')\n"
        "def _bypassed():\n"
        "    return 'should never be reachable in read_only mode'\n"
        "try:\n"
        "    s._audit_registered_tools('read_only')\n"
        "    print('NO_RAISE')\n"
        "except RuntimeError as e:\n"
        "    print('RAISED:' + ('bypassed tool_gate' in str(e)).__str__())\n"
    )
    out = _run_server_subprocess(tmp_path, {"VAULT_ACCESS_MODE": "read_only"}, code)
    assert out.strip().splitlines()[-1] == "RAISED:True"


# --- LOW-7: build_app() access_mode cannot diverge from import-time mode ---

def test_low7_build_app_rejects_mismatched_access_mode(tmp_path):
    code = (
        "import obsidian_vault_mcp.server as s\n"
        "try:\n"
        "    s.build_app('read_only')\n"
        "    print('NO_RAISE')\n"
        "except ValueError as e:\n"
        "    print('RAISED:' + ('does not match' in str(e)).__str__())\n"
    )
    out = _run_server_subprocess(tmp_path, {"VAULT_ACCESS_MODE": "full"}, code)
    assert out.strip().splitlines()[-1] == "RAISED:True"


# --- LOW-9: allowed_hosts derived per-instance from VAULT_BASE_URL ---

def test_low9_allowed_hosts_derived_from_vault_base_url(tmp_path):
    code = (
        "import obsidian_vault_mcp.server as s\n"
        "print(','.join(s._allowed_hosts()))\n"
    )
    out = _run_server_subprocess(
        tmp_path, {"VAULT_BASE_URL": "https://vault-cb-marketing.bensum.org"}, code
    )
    hosts = out.strip().splitlines()[-1].split(",")
    assert "vault-cb-marketing.bensum.org" in hosts
    assert "vault-cb.bensum.org" not in hosts
    assert "vault.bensum.org" not in hosts


def test_low9_allowed_hosts_falls_back_when_base_url_unset(tmp_path):
    code = (
        "import obsidian_vault_mcp.server as s\n"
        "print(','.join(s._allowed_hosts()))\n"
    )
    out = _run_server_subprocess(tmp_path, {}, code)
    hosts = out.strip().splitlines()[-1].split(",")
    assert "vault.bensum.org" in hosts  # old static list preserved as fallback


# --- HIGH-2: canonical vault directory permissions (informational; the real
# fix is host state, not code -- covered live by
# mcp-infrastructure/scripts/cb_brain_marketing_ro_acceptance.py). No test
# here: this repo has no access to the live host's /home/ben_sum/vaults tree
# in CI, and duplicating a filesystem-permission check against a path that
# doesn't exist in this sandbox would be a false sense of coverage.
