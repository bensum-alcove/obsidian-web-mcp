"""Subprocess adapter for the Build Orchestrator authoring contract (bo-authoring-contract-core-v1).

Vault MCP must not re-encode BO schema rules (project/tier/risk-domain enums,
dependency-graph validation, schedule-authority immutability, etc). Instead this
module shells out to `authoring_contract.py`'s JSON stdin/stdout CLI, which lives
in the build-orchestrator repo and is the single executable definition of BO
authoring semantics.

Per the vault-bo-authoring-mcp-v1 spec's architecture constraint: "If the adapter
is absent, wrong version or fails, BO schedule activation fails closed." Every
function here raises BOContractError (never returns a partial/guessed result) on
any of those conditions -- callers (tools/build_orchestrator.py, bo_guard.py) are
responsible for turning that into a fail-closed response.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass

from . import config

logger = logging.getLogger(__name__)

EXPECTED_SCHEMA_VERSION = 6
EXPECTED_CONTRACT_VERSION = "1.0.0"


@dataclass
class BOContractError(Exception):
    """Raised whenever the authoring-contract adapter cannot be trusted.

    `code` is one of: adapter_missing, adapter_timeout, adapter_bad_output,
    adapter_error, version_mismatch. Never raised for a normal validation
    failure (unknown project, dependency cycle, etc) -- those come back as a
    structured `{"ok": False, "errors": [...]}` result, not an exception.
    """

    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def _invoke(payload: dict, timeout: float | None = None) -> dict:
    """Run one JSON request through the authoring_contract.py CLI.

    shell=False, argument vector only -- never string-interpolated into a shell.
    """
    timeout = timeout if timeout is not None else config.BO_AUTHORING_CONTRACT_TIMEOUT_SECONDS
    cmd = [config.BO_AUTHORING_CONTRACT_PYTHON, config.BO_AUTHORING_CONTRACT_PATH]

    try:
        result = subprocess.run(
            cmd,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except FileNotFoundError as e:
        raise BOContractError(
            "adapter_missing",
            f"authoring contract adapter not found (cmd={cmd!r}): {e}",
        ) from e
    except subprocess.TimeoutExpired as e:
        raise BOContractError(
            "adapter_timeout",
            f"authoring contract adapter did not respond within {timeout}s (cmd={cmd!r})",
        ) from e

    stdout = (result.stdout or "").strip()
    try:
        parsed = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError as e:
        raise BOContractError(
            "adapter_bad_output",
            f"authoring contract adapter returned non-JSON output "
            f"(exit={result.returncode}, stderr={result.stderr[:500]!r}): {e}",
        ) from e

    if result.returncode != 0:
        raise BOContractError(
            "adapter_error",
            parsed.get("error") if isinstance(parsed, dict) else
            f"adapter exited {result.returncode}: {result.stderr[:500]!r}",
        )

    if not isinstance(parsed, dict):
        raise BOContractError(
            "adapter_bad_output",
            f"authoring contract adapter returned non-object JSON: {parsed!r}",
        )

    return parsed


def check_version(timeout: float | None = None) -> dict:
    """Call op=version and assert schema_version matches what this adapter was
    built against. Raises BOContractError (version_mismatch) on drift -- this is
    the fail-closed check every BO tool call must pass before doing anything else.
    """
    result = _invoke({"op": "version"}, timeout=timeout)
    schema_version = result.get("schema_version")
    if schema_version != EXPECTED_SCHEMA_VERSION:
        raise BOContractError(
            "version_mismatch",
            f"authoring contract schema_version {schema_version!r} != expected "
            f"{EXPECTED_SCHEMA_VERSION!r} -- refusing to trust a drifted adapter",
        )
    return result


def validate_graph(nodes: list[dict], mode: str = "strict_new", config_override: dict | None = None,
                    timeout: float | None = None) -> dict:
    """Validate a whole proposed build graph. Read-only -- never mutates anything
    (the adapter's own DB preflight lookups are plain SELECTs).

    Returns {"ok": bool, "errors": [...], "warnings": [...]}.
    """
    payload = {"op": "validate_graph", "mode": mode, "nodes": nodes}
    if config_override is not None:
        payload["config"] = config_override
    return _invoke(payload, timeout=timeout)


def render_graph(specs: list[dict], timeout: float | None = None) -> dict:
    """Render spec markdown + schedule-entry YAML for a set of proposed builds,
    via authoring_contract's structured yaml.safe_dump renderers -- never a
    hand-built string. Pure computation; never touches the filesystem.

    Returns {"rendered": {build_id: {"spec": "...", "schedule_entry": "..."}}}.
    """
    return _invoke({"op": "render_graph", "specs": specs}, timeout=timeout)


def preflight_ids(build_ids: list[str], timeout: float | None = None) -> dict:
    """Bulk read-only status lookup for a list of build ids.

    Returns {"results": {build_id: status_or_None}}.
    """
    return _invoke(
        {"op": "preflight", "preflight_op": "ids", "build_ids": build_ids},
        timeout=timeout,
    )


def preflight_schedule_rewrite(schedule_path: str, new_builds: list[dict], mode: str = "strict_new",
                                timeout: float | None = None) -> dict:
    """Preflight for rewriting an existing schedule file's `builds:` list.

    Rejects (in the result's errors, not an exception) dropping/editing a
    dispatched-or-later or terminal entry that's currently bound to
    schedule_path. Read-only -- queries live DB state, never writes.
    """
    return _invoke(
        {
            "op": "preflight",
            "preflight_op": "schedule_rewrite",
            "schedule_path": schedule_path,
            "new_builds": new_builds,
            "mode": mode,
        },
        timeout=timeout,
    )


def preflight_schedule_move(schedule_path: str, timeout: float | None = None) -> dict:
    """Preflight for moving, renaming, or deleting a schedule file outright.

    Rejects (in the result's errors) the operation if any non-terminal build is
    still bound to schedule_path. Read-only.
    """
    return _invoke(
        {"op": "preflight", "preflight_op": "schedule_move", "schedule_path": schedule_path},
        timeout=timeout,
    )
