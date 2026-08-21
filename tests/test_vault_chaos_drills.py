"""Tests for scripts/vault_chaos_drills.py (vault-chaos-recovery-suite build).

Each test runs the real drill function against a fresh tmp_path -- the same
isolation the drill script itself uses when run standalone -- and asserts on
the specific detected/contained/recovered claims, not just the aggregate
`ok` flag, so a regression in one property doesn't hide behind another.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "vault_chaos_drills.py"


@pytest.fixture(scope="module")
def chaos_mod():
    spec = importlib.util.spec_from_file_location("vault_chaos_drills", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec -- the module's @dataclass fields
    # need `sys.modules[cls.__module__]` to resolve during dataclass field
    # type-hint processing (same fix test_vault_functional_canary.py needed).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_semantic_index_loss_detected_and_rebuilt_from_markdown(chaos_mod, tmp_path):
    result = chaos_mod.drill_semantic_index_loss(tmp_path)
    assert result.error is None, result.error
    assert result.detected, "index loss must surface as a 'building' status, not silence"
    assert result.recovered, "build_index() must rebuild the fixture content from Markdown"
    assert result.ok


def test_entity_index_loss_detected_and_rebuilt_from_markdown(chaos_mod, tmp_path):
    result = chaos_mod.drill_entity_index_loss(tmp_path)
    assert result.detected
    assert result.recovered
    assert result.evidence["rebuilt_entity_names"] == ["Acme"]
    assert result.ok


def test_atomic_write_interruption_never_corrupts_original_file(chaos_mod, tmp_path):
    result = chaos_mod.drill_atomic_write_interruption(tmp_path)
    assert result.detected, "the injected os.replace failure must surface to the caller"
    assert result.contained, "original bytes must be provably unchanged after the injected failure"
    assert result.evidence["leftover_tempfiles"] == []
    assert result.recovered
    assert result.ok


def test_read_only_filesystem_blocks_write_without_corruption(chaos_mod, tmp_path):
    result = chaos_mod.drill_read_only_filesystem(tmp_path)
    assert result.detected
    assert result.contained
    assert result.recovered
    assert result.evidence["safety_sweep_flagged_missing_world_read"]
    assert result.evidence["safety_sweep_repaired_permissions"]
    assert result.ok


def test_stale_concurrent_edit_blocked_by_guarded_primitive(chaos_mod, tmp_path):
    result = chaos_mod.drill_stale_concurrent_edit(tmp_path)
    assert result.detected, "the hash-guard pattern must refuse a stale write"
    assert result.contained, "writer B's content must survive writer A's stale attempt"
    assert result.recovered, "a re-read-and-retry must succeed once the writer is no longer stale"


def test_stale_concurrent_edit_confirms_public_mcp_write_tool_gap(chaos_mod, tmp_path):
    """Not a regression test for a bug -- a pinned finding. If this ever
    flips to False, vault_write gained a concurrency guard and
    chaos/failure-matrix.md's residual-gap note needs updating, not this
    assertion silently deleted."""
    result = chaos_mod.drill_stale_concurrent_edit(tmp_path)
    assert result.evidence["live_mcp_write_tool_gap_confirmed"] is True


def test_malformed_yaml_detected_known_signature_repaired_unknown_flagged(chaos_mod, tmp_path):
    result = chaos_mod.drill_malformed_yaml_write(tmp_path)
    assert result.detected
    assert result.contained
    assert result.recovered
    assert result.evidence["alert_open_outcome"] == "new_failure"
    assert result.evidence["alert_close_outcome"] == "recovered"
    assert len(result.evidence["alert_messages_sent"]) == 2
    assert result.ok


def test_malformed_yaml_confirms_live_write_path_has_no_validation(chaos_mod, tmp_path):
    """Pinned finding, same rationale as the concurrent-edit gap test above."""
    result = chaos_mod.drill_malformed_yaml_write(tmp_path)
    assert result.evidence["live_write_path_accepted_malformed_yaml_unvalidated"] is True


def test_bulk_modification_detected_and_recovered_via_git(chaos_mod, tmp_path):
    result = chaos_mod.drill_bulk_modification(tmp_path)
    assert result.detected
    assert result.evidence["changed_file_count"] == 5
    assert result.evidence["alert_open_outcome"] == "new_failure"
    assert result.evidence["alert_close_outcome"] == "recovered"
    assert result.evidence["post_recovery_hashes_match_baseline"]
    assert result.ok


def test_run_all_restores_config_vault_path_afterward(chaos_mod):
    """Structural safety net for the spec's own hard constraint: every drill
    temporarily repoints config.VAULT_PATH at its own throwaway tempdir and
    must restore the prior value afterward -- a real vault path must never
    be left pointing at a deleted chaos tempdir."""
    import obsidian_vault_mcp.config as config_module

    original_vault_path = config_module.VAULT_PATH
    report = chaos_mod.run_all()
    assert report["all_ok"], json.dumps(report, indent=2)
    assert config_module.VAULT_PATH == original_vault_path


def test_run_all_produces_fixed_schema_report(chaos_mod):
    report = chaos_mod.run_all()
    assert set(report.keys()) == {"generated_at", "drills", "all_ok"}
    scenario_ids = sorted(d["scenario_id"] for d in report["drills"])
    assert scenario_ids == [2, 3, 4, 5, 6, 7, 11]
    for drill in report["drills"]:
        assert set(drill.keys()) == {
            "scenario_id", "name", "classification", "ok", "detected",
            "contained", "recovered", "max_loss", "evidence", "error",
        }
