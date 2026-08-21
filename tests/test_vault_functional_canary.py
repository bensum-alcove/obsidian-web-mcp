"""Tests for scripts/vault_functional_canary.py (vault-observability-slo build)."""

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "vault_functional_canary.py"


@pytest.fixture(scope="module")
def canary_mod():
    spec = importlib.util.spec_from_file_location("vault_functional_canary", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec -- the module's @dataclass fields
    # need `sys.modules[cls.__module__]` to resolve during dataclass field
    # type-hint processing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def vault(tmp_path, monkeypatch, canary_mod):
    import obsidian_vault_mcp.config as config
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    # vault_lock keys its lock registry off resolved paths, not VAULT_PATH, so
    # no extra isolation is needed there between tests.
    return tmp_path


def _now():
    return datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)


def test_full_run_all_layers_pass(canary_mod, vault):
    status = canary_mod.run_canary(vault, _now())
    assert status["overall_ok"] is True
    assert status["layers_failing"] == []
    assert [layer["layer"] for layer in status["layers"]] == canary_mod.LAYER_ORDER
    # Cleanup actually happened -- no leftover probe files.
    probes = list((vault / canary_mod.CANARY_DIR).glob("probe-*.md"))
    assert probes == []
    # The immutable fixture was created and is still present.
    assert (vault / canary_mod.IMMUTABLE_CANARY_REL).exists()


def test_second_run_reuses_immutable_canary_without_recreating(canary_mod, vault):
    first = canary_mod.run_canary(vault, _now())
    assert first["layers"][0]["detail"] == "created on first run"
    second = canary_mod.run_canary(vault, _now())
    assert second["overall_ok"] is True
    assert second["layers"][0]["detail"] == ""  # not recreated the second time


def test_drifted_immutable_canary_is_detected_and_downstream_layers_skip(canary_mod, vault):
    canary_mod.run_canary(vault, _now())  # bootstrap the fixture
    resolved = vault / canary_mod.IMMUTABLE_CANARY_REL
    resolved.write_text("---\ntype: canary-fixture\n---\n\nsomeone edited this\n")

    status = canary_mod.run_canary(vault, _now())
    assert status["overall_ok"] is False
    assert "read_immutable_canary" in status["layers_failing"]
    by_layer = {layer["layer"]: layer for layer in status["layers"]}
    assert by_layer["exact_query"]["detail"].startswith("skipped:")
    assert by_layer["hybrid_query"]["detail"].startswith("skipped:")
    # create/patch/cleanup do not depend on the immutable fixture -- they still run.
    assert by_layer["create_scratch"]["ok"] is True


def test_concurrent_modification_blocks_the_patch(canary_mod, vault):
    run_id = "conctest"
    result, rel_path, revision = canary_mod.create_scratch(vault, run_id)
    assert result.ok
    # Simulate a concurrent writer landing between create_scratch and patch.
    (vault / rel_path).write_text("---\ntype: canary-fixture\nphase: tampered\n---\n\nx\n")

    patch_result = canary_mod.patch_with_expected_revision(vault, rel_path, revision, run_id)
    assert patch_result.ok is False
    assert "ConcurrentModificationError" in patch_result.detail
    # The tampering writer's content must be left untouched -- the guard
    # never overwrites on a hash mismatch.
    assert "tampered" in (vault / rel_path).read_text()


def test_never_touches_paths_outside_scratch_namespace(canary_mod, vault):
    (vault / "real-knowledge.md").write_text("---\ntype: note\n---\n\nDo not touch.\n")
    before = (vault / "real-knowledge.md").read_text()

    canary_mod.run_canary(vault, _now())

    after = (vault / "real-knowledge.md").read_text()
    assert before == after
    # Nothing outside _scratch/canary/ was created either.
    all_paths = {str(p.relative_to(vault)) for p in vault.rglob("*") if p.is_file()}
    non_scratch = {p for p in all_paths if not p.startswith(canary_mod.CANARY_DIR + "/")}
    assert non_scratch == {"real-knowledge.md"}


def test_scratch_scope_guard_fails_closed_on_a_synthetic_violation(canary_mod, vault):
    result = canary_mod.assert_scratch_scope(["_scratch/canary/ok.md", "SYSTEM-FACTS.md"])
    assert result.ok is False
    assert "SYSTEM-FACTS.md" in result.detail


def test_main_cli_vault_path_override_stays_consistent_with_config_global(
    canary_mod, tmp_path, monkeypatch
):
    """Regression test: hybrid_query() calls the shared vault_search tool,
    which resolves against config.VAULT_PATH (a global), not the vault_path
    parameter threaded through the rest of this script. An explicit
    --vault-path CLI override must keep that global in sync, or hybrid_query
    silently fails against whatever config.VAULT_PATH happened to default to
    (caught by an end-to-end CLI smoke test, not the mocked unit tests
    above, which coincidentally kept both in sync via monkeypatch)."""
    import obsidian_vault_mcp.config as config
    other_vault = tmp_path / "unrelated-default-vault"
    other_vault.mkdir()
    monkeypatch.setattr(config, "VAULT_PATH", other_vault)  # simulates a stale default

    target_vault = tmp_path / "explicit-target-vault"
    target_vault.mkdir()
    status_dir = tmp_path / "status-out"

    monkeypatch.setattr(sys, "argv", [
        "vault_functional_canary.py",
        "--vault-path", str(target_vault),
        "--vault-name", "explicit-target",
        "--status-dir", str(status_dir),
    ])
    exit_code = canary_mod.main()
    assert exit_code == 0
    status = json.loads((status_dir / "canary-explicit-target.json").read_text())
    assert status["overall_ok"] is True
    assert status["layers_failing"] == []
    assert config.VAULT_PATH == target_vault


def test_write_status_is_valid_json_and_atomic(canary_mod, vault, tmp_path):
    status_dir = tmp_path / "status-out"
    status = canary_mod.run_canary(vault, _now())
    status["vault_name"] = "test-vault"
    path = canary_mod.write_status(status, "test-vault", status_dir)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["overall_ok"] is True
    # No leftover tempfile from the atomic write.
    assert list(status_dir.glob("*.tmp")) == []
