"""Tests for bo_guard.py -- the shadow-only BO path-mutation guard.

Deployed shadow-only in vault-bo-authoring-mcp-v1 (BO_PATH_GUARD_MODE defaults
to "shadow"), but the enforce path must already be correct and tested per the
spec ("The future enforce behavior must already be implemented and tested...
Do NOT turn enforce on in this build"). bo_contract calls are monkeypatched
throughout so these tests never touch the real adapter/DB -- they test THIS
module's mode-gating, path-scoping, and issue-to-block mapping.
"""

import pytest

from obsidian_vault_mcp import bo_contract, bo_guard
from obsidian_vault_mcp.write_contract import PathMutationContext, WriteContext

SCHEDULE_PATH = "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml"
SPEC_PATH = "Personal/Build Orchestrator/specs/scratch-build.md"

VALID_SCHEDULE_OLD = (
    "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# scratch\n\nbuilds:\n\n"
    "  - id: existing-build\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
    "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/existing-build.md\n"
)
VALID_SCHEDULE_NEW = VALID_SCHEDULE_OLD + (
    "  - id: new-build\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
    "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/new-build.md\n"
)


@pytest.fixture(autouse=True)
def _reset_mode(monkeypatch):
    monkeypatch.delenv("BO_PATH_GUARD_MODE", raising=False)


def test_mode_defaults_to_shadow(monkeypatch):
    assert bo_guard.get_mode() == "shadow"


def test_unrecognised_mode_falls_back_to_shadow(monkeypatch):
    monkeypatch.setenv("BO_PATH_GUARD_MODE", "bogus")
    assert bo_guard.get_mode() == "shadow"


def test_off_mode_skips_evaluation_entirely(monkeypatch):
    monkeypatch.setenv("BO_PATH_GUARD_MODE", "off")

    def boom(*a, **k):
        raise AssertionError("adapter should never be called in off mode")

    monkeypatch.setattr(bo_contract, "preflight_schedule_rewrite", boom)
    result = bo_guard.evaluate_content(
        WriteContext(path=SCHEDULE_PATH, old_content=VALID_SCHEDULE_OLD, new_content=VALID_SCHEDULE_NEW, tool="vault_write")
    )
    assert result.issues == []
    assert not result.blocked


def test_non_bo_path_never_touches_adapter(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("adapter should never be called for a non-BO path")

    monkeypatch.setattr(bo_contract, "preflight_schedule_rewrite", boom)
    monkeypatch.setattr(bo_contract, "preflight_ids", boom)
    result = bo_guard.evaluate_content(
        WriteContext(path="Clients/some-note.md", old_content="a", new_content="b", tool="vault_write")
    )
    assert result.issues == []


def test_new_schedule_file_skips_rewrite_check(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("nothing is bound to a schedule that doesn't exist yet")

    monkeypatch.setattr(bo_contract, "preflight_schedule_rewrite", boom)
    result = bo_guard.evaluate_content(
        WriteContext(path=SCHEDULE_PATH, old_content=None, new_content=VALID_SCHEDULE_NEW, tool="vault_write")
    )
    assert result.issues == []


def test_noop_write_skips_rewrite_check(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a byte-identical write should never trigger a preflight call")

    monkeypatch.setattr(bo_contract, "preflight_schedule_rewrite", boom)
    result = bo_guard.evaluate_content(
        WriteContext(path=SCHEDULE_PATH, old_content=VALID_SCHEDULE_OLD, new_content=VALID_SCHEDULE_OLD, tool="vault_write")
    )
    assert result.issues == []


def test_schedule_rewrite_orphaning_reported_in_shadow_but_not_blocked(monkeypatch):
    monkeypatch.setattr(
        bo_contract, "preflight_schedule_rewrite",
        lambda schedule_path, new_builds, mode="strict_new", timeout=None: {
            "ok": False,
            "errors": [{"code": "schedule_binding_orphaned", "message": "would drop existing-build", "build_id": "existing-build"}],
            "warnings": [],
        },
    )
    monkeypatch.setenv("BO_PATH_GUARD_MODE", "shadow")
    # new_content drops the existing-build entry entirely -- an orphaning rewrite
    dropped = "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# scratch\n\nbuilds:\n\n" + \
        "  - id: only-new\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n" + \
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/only-new.md\n"
    result = bo_guard.evaluate_content(
        WriteContext(path=SCHEDULE_PATH, old_content=VALID_SCHEDULE_OLD, new_content=dropped, tool="vault_write")
    )
    assert len(result.issues) == 1
    assert result.issues[0].rule_id == "bo-guard-schedule-rewrite"
    assert not result.blocked  # shadow never blocks
    # enforce() must not raise in shadow mode
    bo_guard.enforce(WriteContext(path=SCHEDULE_PATH, old_content=VALID_SCHEDULE_OLD, new_content=dropped, tool="vault_write"))


def test_schedule_rewrite_orphaning_blocks_in_enforce_mode(monkeypatch):
    monkeypatch.setattr(
        bo_contract, "preflight_schedule_rewrite",
        lambda schedule_path, new_builds, mode="strict_new", timeout=None: {
            "ok": False,
            "errors": [{"code": "schedule_binding_orphaned", "message": "would drop existing-build"}],
            "warnings": [],
        },
    )
    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    dropped = "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# scratch\n\nbuilds:\n\n" + \
        "  - id: only-new\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n" + \
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/only-new.md\n"
    with pytest.raises(bo_guard.BOGuardError):
        bo_guard.enforce(
            WriteContext(path=SCHEDULE_PATH, old_content=VALID_SCHEDULE_OLD, new_content=dropped, tool="vault_write")
        )


def test_valid_schedule_append_has_no_issues(monkeypatch):
    monkeypatch.setattr(
        bo_contract, "preflight_schedule_rewrite",
        lambda schedule_path, new_builds, mode="strict_new", timeout=None: {"ok": True, "errors": [], "warnings": []},
    )
    result = bo_guard.evaluate_content(
        WriteContext(path=SCHEDULE_PATH, old_content=VALID_SCHEDULE_OLD, new_content=VALID_SCHEDULE_NEW, tool="vault_write")
    )
    assert result.issues == []


def test_unparseable_new_schedule_content_reports_advisory_not_reject(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not call the adapter when the content itself can't be parsed")

    monkeypatch.setattr(bo_contract, "preflight_schedule_rewrite", boom)
    result = bo_guard.evaluate_content(
        WriteContext(path=SCHEDULE_PATH, old_content=VALID_SCHEDULE_OLD, new_content="not: [valid, yaml:::", tool="vault_write")
    )
    assert len(result.issues) == 1
    assert result.issues[0].severity == "advisory"


def test_adapter_unavailable_during_schedule_rewrite_is_reject_but_shadow_never_blocks(monkeypatch):
    def boom(*a, **k):
        raise bo_contract.BOContractError("adapter_missing", "not found")

    monkeypatch.setattr(bo_contract, "preflight_schedule_rewrite", boom)
    result = bo_guard.evaluate_content(
        WriteContext(path=SCHEDULE_PATH, old_content=VALID_SCHEDULE_OLD, new_content=VALID_SCHEDULE_NEW, tool="vault_write")
    )
    assert result.issues[0].rule_id == "bo-guard-adapter-unavailable"
    assert result.issues[0].severity == "reject"
    assert not result.blocked  # mode is shadow (default)


def test_spec_rewrite_of_dispatched_build_blocked_in_enforce(monkeypatch):
    monkeypatch.setattr(
        bo_contract, "preflight_ids",
        lambda build_ids, timeout=None: {"results": {"scratch-build": "dispatched"}},
    )
    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    with pytest.raises(bo_guard.BOGuardError):
        bo_guard.enforce(WriteContext(path=SPEC_PATH, old_content="old body", new_content="new body", tool="vault_write"))


def test_spec_rewrite_of_proposed_build_allowed(monkeypatch):
    monkeypatch.setattr(
        bo_contract, "preflight_ids",
        lambda build_ids, timeout=None: {"results": {"scratch-build": "proposed"}},
    )
    result = bo_guard.evaluate_content(
        WriteContext(path=SPEC_PATH, old_content="old body", new_content="new body", tool="vault_write")
    )
    assert result.issues == []


def test_spec_rewrite_of_never_ingested_build_allowed(monkeypatch):
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"scratch-build": None}})
    result = bo_guard.evaluate_content(
        WriteContext(path=SPEC_PATH, old_content="old body", new_content="new body", tool="vault_write")
    )
    assert result.issues == []


def test_new_spec_file_skips_status_check(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a brand-new spec has no binding to check")

    monkeypatch.setattr(bo_contract, "preflight_ids", boom)
    result = bo_guard.evaluate_content(
        WriteContext(path=SPEC_PATH, old_content=None, new_content="new body", tool="vault_write")
    )
    assert result.issues == []


def test_schedule_move_blocked_when_nonterminal_binding_exists(monkeypatch):
    monkeypatch.setattr(
        bo_contract, "preflight_schedule_move",
        lambda schedule_path, timeout=None: {
            "ok": False,
            "errors": [{"code": "schedule_move_would_orphan_binding", "message": "x still bound", "build_id": "x"}],
            "warnings": [],
        },
    )
    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    with pytest.raises(bo_guard.BOGuardError):
        bo_guard.enforce_path_mutation(PathMutationContext(path=SCHEDULE_PATH, operation="delete"))


def test_schedule_move_allowed_when_no_bindings(monkeypatch):
    monkeypatch.setattr(
        bo_contract, "preflight_schedule_move",
        lambda schedule_path, timeout=None: {"ok": True, "errors": [], "warnings": []},
    )
    result = bo_guard.evaluate_path_mutation(PathMutationContext(path=SCHEDULE_PATH, operation="move", destination="Personal/Build Orchestrator/schedules/renamed.yaml"))
    assert result.issues == []


def test_spec_move_of_terminal_build_blocked_in_enforce(monkeypatch):
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"scratch-build": "done"}})
    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    with pytest.raises(bo_guard.BOGuardError):
        bo_guard.enforce_path_mutation(PathMutationContext(path=SPEC_PATH, operation="delete"))


def test_non_bo_path_mutation_never_touches_adapter(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("adapter should never be called for a non-BO path")

    monkeypatch.setattr(bo_contract, "preflight_schedule_move", boom)
    monkeypatch.setattr(bo_contract, "preflight_ids", boom)
    result = bo_guard.evaluate_path_mutation(PathMutationContext(path="Clients/some-note.md", operation="delete"))
    assert result.issues == []
