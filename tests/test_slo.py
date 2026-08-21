"""Tests for obsidian_vault_mcp.slo (vault-observability-slo build)."""

from obsidian_vault_mcp.slo import SLI, Direction, Status, REGISTRY, layers


def test_registry_is_populated_with_documented_signals():
    expected_ids = {
        "mcp_availability", "functional_read_query", "index_freshness_seconds",
        "backup_age_hours", "restore_drill_age_days", "retrieval_trend",
        "validation_rejects_count", "concurrency_conflicts_count",
        "malformed_notes_count", "contradiction_count", "dreaming_state",
        "hot_md_policy_state",
    }
    assert expected_ids <= set(REGISTRY)


def test_every_sli_has_a_non_empty_baseline_evidence_note():
    """Every threshold must be traceable to either real evidence or an
    explicit 'new signal' admission -- never a bare unexplained number."""
    for sli in REGISTRY.values():
        assert sli.baseline_evidence, f"{sli.id} has no baseline_evidence"


def test_direction_above_ok_warning_critical_boundaries():
    sli = SLI(id="x", description="", layer="l", unit="h", direction=Direction.ABOVE,
              warning=24, critical=48)
    assert sli.evaluate(10) == Status.OK
    assert sli.evaluate(24) == Status.WARNING
    assert sli.evaluate(47.9) == Status.WARNING
    assert sli.evaluate(48) == Status.CRITICAL
    assert sli.evaluate(100) == Status.CRITICAL


def test_direction_below_ok_warning_critical_boundaries():
    sli = SLI(id="x", description="", layer="l", unit="score", direction=Direction.BELOW,
              warning=-0.05, critical=-0.15)
    assert sli.evaluate(0.1) == Status.OK
    assert sli.evaluate(-0.05) == Status.WARNING
    assert sli.evaluate(-0.15) == Status.CRITICAL
    assert sli.evaluate(-0.5) == Status.CRITICAL


def test_direction_equals_ok_value_vs_anything_else():
    sli = SLI(id="x", description="", layer="l", unit="bool", direction=Direction.EQUALS,
              warning=None, critical=None, ok_value="completed")
    assert sli.evaluate("completed") == Status.OK
    assert sli.evaluate("missed") == Status.CRITICAL
    assert sli.evaluate(None) == Status.UNKNOWN


def test_none_value_is_always_unknown_never_ok():
    for sli in REGISTRY.values():
        assert sli.evaluate(None) == Status.UNKNOWN


def test_non_numeric_value_on_a_numeric_sli_is_unknown_not_a_crash():
    sli = SLI(id="x", description="", layer="l", unit="h", direction=Direction.ABOVE,
              warning=24, critical=48)
    assert sli.evaluate("not-a-number") == Status.UNKNOWN


def test_layers_groups_every_registered_sli():
    all_layers = layers()
    covered = {sli.layer for sli in REGISTRY.values()}
    assert set(all_layers) == covered
