"""Guard the `mcp` library pin: the server imports mcp.server.fastmcp, removed in mcp 2.x."""

import importlib
import tomllib
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _major(version: str) -> int:
    return int(version.split(".")[0])


def test_installed_mcp_is_1x():
    assert _major(metadata.version("mcp")) == 1


def test_fastmcp_importable():
    module = importlib.import_module("mcp.server.fastmcp")
    assert hasattr(module, "FastMCP")


def test_pyproject_caps_mcp_below_2():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    mcp_specs = [d.replace(" ", "") for d in deps if d.startswith("mcp")]
    assert mcp_specs and all(",<2" in s for s in mcp_specs), mcp_specs


def test_uv_lock_resolves_mcp_1x():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions = [p["version"] for p in lock["package"] if p["name"] == "mcp"]
    assert versions, "mcp missing from uv.lock"
    assert all(_major(v) == 1 for v in versions), versions
