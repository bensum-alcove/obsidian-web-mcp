"""Tests for bo_contract.py -- the authoring-contract subprocess adapter.

Adapter mechanics (timeout, missing binary, bad output, nonzero exit, version
drift) are tested hermetically against a monkeypatched subprocess.run, since
those failure modes are about THIS module's fail-closed behaviour, not about
authoring_contract.py's own correctness (795 tests in build-orchestrator's own
suite already cover that). A couple of real end-to-end calls against the real
CLI are included, skipped automatically if that repo isn't present on this
host, to prove the wiring actually works against the real thing.
"""

import json
import subprocess
from pathlib import Path

import pytest

from obsidian_vault_mcp import bo_contract, config


class FakeCompletedProcess:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_invoke_success(monkeypatch):
    monkeypatch.setattr(
        bo_contract.subprocess, "run",
        lambda *a, **k: FakeCompletedProcess(stdout='{"ok": true, "schema_version": 6, "contract_version": "1.0.0"}'),
    )
    result = bo_contract._invoke({"op": "version"})
    assert result == {"ok": True, "schema_version": 6, "contract_version": "1.0.0"}


def test_invoke_uses_shell_false_and_argv(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeCompletedProcess(stdout='{"ok": true}')

    monkeypatch.setattr(bo_contract.subprocess, "run", fake_run)
    bo_contract._invoke({"op": "version"})
    assert captured["kwargs"]["shell"] is False
    assert captured["cmd"] == [config.BO_AUTHORING_CONTRACT_PYTHON, config.BO_AUTHORING_CONTRACT_PATH]
    assert isinstance(captured["cmd"], list)


def test_invoke_missing_adapter(monkeypatch):
    def fake_run(*a, **k):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(bo_contract.subprocess, "run", fake_run)
    with pytest.raises(bo_contract.BOContractError) as exc:
        bo_contract._invoke({"op": "version"})
    assert exc.value.code == "adapter_missing"


def test_invoke_timeout(monkeypatch):
    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(bo_contract.subprocess, "run", fake_run)
    with pytest.raises(bo_contract.BOContractError) as exc:
        bo_contract._invoke({"op": "version"})
    assert exc.value.code == "adapter_timeout"


def test_invoke_bad_json_output(monkeypatch):
    monkeypatch.setattr(bo_contract.subprocess, "run", lambda *a, **k: FakeCompletedProcess(stdout="not json"))
    with pytest.raises(bo_contract.BOContractError) as exc:
        bo_contract._invoke({"op": "version"})
    assert exc.value.code == "adapter_bad_output"


def test_invoke_non_object_json(monkeypatch):
    monkeypatch.setattr(bo_contract.subprocess, "run", lambda *a, **k: FakeCompletedProcess(stdout="[1, 2, 3]"))
    with pytest.raises(bo_contract.BOContractError) as exc:
        bo_contract._invoke({"op": "version"})
    assert exc.value.code == "adapter_bad_output"


def test_invoke_nonzero_exit_with_parseable_error(monkeypatch):
    monkeypatch.setattr(
        bo_contract.subprocess, "run",
        lambda *a, **k: FakeCompletedProcess(stdout='{"ok": false, "error": "unknown op"}', returncode=1),
    )
    with pytest.raises(bo_contract.BOContractError) as exc:
        bo_contract._invoke({"op": "bogus"})
    assert exc.value.code == "adapter_error"
    assert "unknown op" in exc.value.message


def test_invoke_nonzero_exit_unparseable(monkeypatch):
    monkeypatch.setattr(
        bo_contract.subprocess, "run",
        lambda *a, **k: FakeCompletedProcess(stdout="", stderr="traceback...", returncode=1),
    )
    with pytest.raises(bo_contract.BOContractError) as exc:
        bo_contract._invoke({"op": "version"})
    assert exc.value.code == "adapter_error"


def test_check_version_matches(monkeypatch):
    monkeypatch.setattr(
        bo_contract.subprocess, "run",
        lambda *a, **k: FakeCompletedProcess(stdout=json.dumps({"ok": True, "schema_version": 6, "contract_version": "1.0.0"})),
    )
    result = bo_contract.check_version()
    assert result["schema_version"] == 6


def test_check_version_mismatch_fails_closed(monkeypatch):
    monkeypatch.setattr(
        bo_contract.subprocess, "run",
        lambda *a, **k: FakeCompletedProcess(stdout=json.dumps({"ok": True, "schema_version": 4, "contract_version": "0.1.0"})),
    )
    with pytest.raises(bo_contract.BOContractError) as exc:
        bo_contract.check_version()
    assert exc.value.code == "version_mismatch"


def test_validate_graph_passes_payload_shape(monkeypatch):
    captured = {}

    def fake_run(cmd, input=None, **kwargs):
        captured["payload"] = json.loads(input)
        return FakeCompletedProcess(stdout='{"ok": true, "errors": [], "warnings": []}')

    monkeypatch.setattr(bo_contract.subprocess, "run", fake_run)
    result = bo_contract.validate_graph([{"build_id": "x"}], mode="compat_existing")
    assert result["ok"] is True
    assert captured["payload"]["op"] == "validate_graph"
    assert captured["payload"]["mode"] == "compat_existing"
    assert captured["payload"]["nodes"] == [{"build_id": "x"}]
    assert "config" not in captured["payload"]


def test_preflight_schedule_move_payload_shape(monkeypatch):
    captured = {}

    def fake_run(cmd, input=None, **kwargs):
        captured["payload"] = json.loads(input)
        return FakeCompletedProcess(stdout='{"ok": true, "errors": [], "warnings": []}')

    monkeypatch.setattr(bo_contract.subprocess, "run", fake_run)
    bo_contract.preflight_schedule_move("Personal/Build Orchestrator/schedules/x.yaml")
    assert captured["payload"] == {
        "op": "preflight", "preflight_op": "schedule_move",
        "schedule_path": "Personal/Build Orchestrator/schedules/x.yaml",
    }


# --- real end-to-end smoke tests against the actual CLI, if present ---

_ADAPTER_PRESENT = Path(config.BO_AUTHORING_CONTRACT_PATH).exists()


@pytest.mark.skipif(not _ADAPTER_PRESENT, reason="build-orchestrator authoring_contract.py not present on this host")
def test_real_check_version():
    result = bo_contract.check_version()
    assert result["schema_version"] == bo_contract.EXPECTED_SCHEMA_VERSION


@pytest.mark.skipif(not _ADAPTER_PRESENT, reason="build-orchestrator authoring_contract.py not present on this host")
def test_real_validate_graph_rejects_unknown_project():
    node = {
        "build_id": "vault-bo-authoring-mcp-v1-test-node-does-not-exist",
        "schedule_entry": {
            "id": "vault-bo-authoring-mcp-v1-test-node-does-not-exist",
            "title": "t", "description": "t", "run_when": "x", "tier": "simple",
            "depends_on": [], "spec_path": "Personal/Build Orchestrator/specs/vault-bo-authoring-mcp-v1-test-node-does-not-exist.md",
        },
        "spec_markdown": (
            "---\nbuild_id: vault-bo-authoring-mcp-v1-test-node-does-not-exist\ntier: simple\n"
            "project: totally-unconfigured-project-xyz\nstatus: ready\nrisk_domain: observability\n---\n\n"
            "# test\n\nWrite a summary to /tmp/cc-summary-vault-bo-authoring-mcp-v1-test-node-does-not-exist.txt "
            "where the FIRST LINE is exactly: vault-bo-authoring-mcp-v1-test-node-does-not-exist.\n"
        ),
        "schedule_path": "Personal/Build Orchestrator/schedules/does-not-exist.yaml",
    }
    result = bo_contract.validate_graph([node], mode="strict_new")
    assert result["ok"] is False
    assert any(e["code"] == "unknown_project" for e in result["errors"])
