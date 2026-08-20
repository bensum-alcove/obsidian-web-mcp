"""Tests for bo_guard.py -- the shadow-only BO path-mutation guard.

Deployed shadow-only (BO_PATH_GUARD_MODE defaults to "shadow"), but the
enforce path must already be correct and tested. bo_contract calls are
monkeypatched throughout so these tests never touch the real adapter/DB --
they test THIS module's mode-gating, path-scoping, and issue-to-block
mapping.

codex-review-bo-authoring-contract-v1 (2026-08-17), B1, found the first cut
fails open in several ways. Several tests below (marked) previously asserted
the OLD, now-fixed, fail-open behaviour -- they have been rewritten to assert
the closed behaviour instead.
"""

import pytest

from obsidian_vault_mcp import bo_contract, bo_guard
from obsidian_vault_mcp.write_contract import PathMutationContext, WriteContext

SCHEDULE_PATH = "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml"
SPEC_PATH = "Personal/Build Orchestrator/specs/scratch-build.md"

# Captured at collection time, before any per-test monkeypatching -- lets a
# test restore the REAL _freely_mutable_statuses over the autouse stub below
# to exercise its own (no-fallback) fail-closed behaviour directly.
_REAL_FREELY_MUTABLE_STATUSES = bo_guard._freely_mutable_statuses

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


@pytest.fixture(autouse=True)
def _stub_graph_validation_ok(monkeypatch):
    """Most tests here exercise ONE specific rule in isolation; stub the
    whole-graph validator to a clean pass by default so a test only needs to
    override it when the whole-graph check itself is what's under test."""
    monkeypatch.setattr(
        bo_contract, "validate_graph",
        lambda nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None:
            {"ok": True, "errors": [], "warnings": []},
    )


@pytest.fixture(autouse=True)
def _stub_check_version_ok(monkeypatch):
    """Every guard evaluation now asserts the adapter's version fresh (HI-1)
    -- stub it to a matching version by default so tests that don't care
    about version drift stay hermetic (never touch the real adapter
    subprocess) and existing assertions are unaffected. Tests that DO care
    override this."""
    monkeypatch.setattr(
        bo_contract, "check_version",
        lambda timeout=None: {
            "ok": True,
            "schema_version": bo_contract.EXPECTED_SCHEMA_VERSION,
            "contract_version": bo_contract.EXPECTED_CONTRACT_VERSION,
        },
    )


@pytest.fixture(autouse=True)
def _stub_no_source_schedule(monkeypatch):
    """Most spec-rewrite tests exercise the spec's own content rules, not
    the B7 schedule-graph-revalidation path; default every build id to "not
    yet bound to any schedule" so that path is a no-op unless a test
    overrides it."""
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {bid: None for bid in build_ids}},
    )


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
    monkeypatch.setattr(bo_contract, "validate_graph", boom)
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


def test_new_schedule_file_is_now_validated_not_exempt(monkeypatch):
    """B1: brand-new schedules used to short-circuit to `return []` before
    ever calling the adapter. A new schedule must now be validated -- the
    bound-row-preservation preflight has nothing to preserve (no error from
    it), but the whole-graph validator IS called and its findings surface."""
    monkeypatch.setattr(
        bo_contract, "preflight_schedule_rewrite",
        lambda schedule_path, new_builds, mode="strict_new", timeout=None: {"ok": True, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        bo_contract, "validate_graph",
        lambda nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None: {
            "ok": False,
            "errors": [{"code": "unknown_project", "message": "no such project", "build_id": "new-build"}],
            "warnings": [],
        },
    )
    result = bo_guard.evaluate_content(
        WriteContext(path=SCHEDULE_PATH, old_content=None, new_content=VALID_SCHEDULE_NEW, tool="vault_write")
    )
    assert any(i.rule_id == "bo-guard-schedule-graph" for i in result.issues)


def test_new_schedule_file_with_valid_content_has_no_issues(monkeypatch):
    monkeypatch.setattr(
        bo_contract, "preflight_schedule_rewrite",
        lambda schedule_path, new_builds, mode="strict_new", timeout=None: {"ok": True, "errors": [], "warnings": []},
    )
    result = bo_guard.evaluate_content(
        WriteContext(path=SCHEDULE_PATH, old_content=None, new_content=VALID_SCHEDULE_NEW, tool="vault_write")
    )
    assert result.issues == []


def test_noop_write_skips_rewrite_check(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a byte-identical write should never trigger a preflight call")

    monkeypatch.setattr(bo_contract, "preflight_schedule_rewrite", boom)
    monkeypatch.setattr(bo_contract, "validate_graph", boom)
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
    assert any(i.rule_id == "bo-guard-schedule-rewrite" for i in result.issues)
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


def test_whole_graph_validation_runs_for_a_schedule_rewrite(monkeypatch):
    """B2: the resulting graph (not just bound-row preservation) must be
    validated for every rewrite, including mixed-project/duplicate-id-style
    findings surfaced only by the whole-graph validator."""
    monkeypatch.setattr(
        bo_contract, "preflight_schedule_rewrite",
        lambda schedule_path, new_builds, mode="strict_new", timeout=None: {"ok": True, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        bo_contract, "validate_graph",
        lambda nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None: {
            "ok": False,
            "errors": [{"code": "mixed_project_schedule", "message": "mixed projects", "path": SCHEDULE_PATH}],
            "warnings": [],
        },
    )
    result = bo_guard.evaluate_content(
        WriteContext(path=SCHEDULE_PATH, old_content=VALID_SCHEDULE_OLD, new_content=VALID_SCHEDULE_NEW, tool="vault_write")
    )
    assert any(i.rule_id == "bo-guard-schedule-graph" for i in result.issues)


def test_unparseable_new_schedule_content_is_reject_not_advisory(monkeypatch):
    """B1: an unparseable/malformed rewrite used to be advisory-only, meaning
    an enforce-mode future build could never actually block it. It's now a
    real reject-severity finding."""
    def boom(*a, **k):
        raise AssertionError("should not call the adapter when the content itself can't be parsed")

    monkeypatch.setattr(bo_contract, "preflight_schedule_rewrite", boom)
    monkeypatch.setattr(bo_contract, "validate_graph", boom)
    result = bo_guard.evaluate_content(
        WriteContext(path=SCHEDULE_PATH, old_content=VALID_SCHEDULE_OLD, new_content="not: [valid, yaml:::", tool="vault_write")
    )
    assert len(result.issues) == 1
    assert result.issues[0].severity == "reject"

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    with pytest.raises(bo_guard.BOGuardError):
        bo_guard.enforce(
            WriteContext(path=SCHEDULE_PATH, old_content=VALID_SCHEDULE_OLD, new_content="not: [valid, yaml:::", tool="vault_write")
        )


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


@pytest.fixture(autouse=True)
def _stub_freely_mutable_statuses(monkeypatch):
    """Isolate spec-status checks from the real adapter subprocess/cache --
    tests that care about a specific status set it explicitly."""
    monkeypatch.setattr(bo_guard, "_freely_mutable_statuses", lambda: {"proposed", "pending", "ready"})


@pytest.fixture(autouse=True)
def _stub_spec_validate_ok(monkeypatch):
    monkeypatch.setattr(
        bo_contract, "preflight_spec_validate",
        lambda build_id, spec_markdown, spec_path, mode="compat_existing", timeout=None: {"ok": True, "errors": []},
    )


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


def test_spec_rewrite_of_freely_mutable_build_with_invalid_content_is_rejected(monkeypatch):
    """B1: a pre-dispatch spec's status being freely-mutable used to be
    sufficient to allow ANY content through. Its own frontmatter/identity/
    tier/risk shape is now checked too."""
    monkeypatch.setattr(
        bo_contract, "preflight_ids",
        lambda build_ids, timeout=None: {"results": {"scratch-build": "proposed"}},
    )
    monkeypatch.setattr(
        bo_contract, "preflight_spec_validate",
        lambda build_id, spec_markdown, spec_path, mode="compat_existing", timeout=None: {
            "ok": False,
            "errors": [{"code": "identity_mismatch", "message": "build_id mismatch", "build_id": build_id}],
        },
    )
    result = bo_guard.evaluate_content(
        WriteContext(path=SPEC_PATH, old_content="old body", new_content="new body", tool="vault_write")
    )
    assert any(i.rule_id == "bo-guard-spec-rewrite" for i in result.issues)


def test_new_spec_file_is_now_content_validated_not_exempt(monkeypatch):
    """B1: brand-new specs used to short-circuit to `return []`. A new spec's
    content is now checked (it has no status binding to check, but its shape
    is validated)."""
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"scratch-build": None}})
    monkeypatch.setattr(
        bo_contract, "preflight_spec_validate",
        lambda build_id, spec_markdown, spec_path, mode="compat_existing", timeout=None: {
            "ok": False,
            "errors": [{"code": "spec_missing_field", "message": "missing tier", "build_id": build_id}],
        },
    )
    result = bo_guard.evaluate_content(
        WriteContext(path=SPEC_PATH, old_content=None, new_content="new body", tool="vault_write")
    )
    assert any(i.rule_id == "bo-guard-spec-rewrite" for i in result.issues)


def test_new_spec_file_with_valid_content_has_no_issues(monkeypatch):
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"scratch-build": None}})
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


# --------------------------------------------------------------------------
# B1 fixes: recursive ancestor/directory protection + inbound-move validation
# --------------------------------------------------------------------------


def test_is_bo_path_matches_exact_directory_and_ancestors():
    assert bo_guard._is_bo_path("Personal/Build Orchestrator/schedules")
    assert bo_guard._is_bo_path("Personal/Build Orchestrator/schedules/")
    assert bo_guard._is_bo_path("Personal/Build Orchestrator")
    assert bo_guard._is_bo_path("Personal")
    assert bo_guard._is_bo_path("Personal/Build Orchestrator/schedules/2026-w1.yaml")
    assert not bo_guard._is_bo_path("Personal/Build Orchestrator/other-file.md")
    assert not bo_guard._is_bo_path("Clients/some-note.md")


def test_moving_schedules_directory_itself_is_evaluated_against_every_file_on_disk(monkeypatch, vault_dir):
    schedules_dir = vault_dir / "Personal" / "Build Orchestrator" / "schedules"
    schedules_dir.mkdir(parents=True)
    (schedules_dir / "2026-w1.yaml").write_text(VALID_SCHEDULE_OLD)
    (schedules_dir / "2026-w2.yaml").write_text(VALID_SCHEDULE_OLD)

    seen_paths = []

    def fake_preflight_schedule_move(schedule_path, timeout=None):
        seen_paths.append(schedule_path)
        return {"ok": True, "errors": [], "warnings": []}

    monkeypatch.setattr(bo_contract, "preflight_schedule_move", fake_preflight_schedule_move)
    monkeypatch.setenv("BO_PATH_GUARD_MODE", "shadow")

    result = bo_guard.evaluate_path_mutation(
        PathMutationContext(
            path="Personal/Build Orchestrator/schedules", operation="move",
            destination="Personal/Build Orchestrator/schedules-renamed",
        )
    )
    assert result.mode == "shadow"
    assert set(seen_paths) == {
        "Personal/Build Orchestrator/schedules/2026-w1.yaml",
        "Personal/Build Orchestrator/schedules/2026-w2.yaml",
    }


def test_moving_ancestor_of_schedules_directory_is_still_evaluated(monkeypatch, vault_dir):
    schedules_dir = vault_dir / "Personal" / "Build Orchestrator" / "schedules"
    schedules_dir.mkdir(parents=True)
    (schedules_dir / "2026-w1.yaml").write_text(VALID_SCHEDULE_OLD)

    calls = []
    monkeypatch.setattr(
        bo_contract, "preflight_schedule_move",
        lambda schedule_path, timeout=None: (calls.append(schedule_path), {"ok": True, "errors": [], "warnings": []})[1],
    )
    result = bo_guard.evaluate_path_mutation(
        PathMutationContext(path="Personal/Build Orchestrator", operation="move", destination="Personal/BO-renamed")
    )
    assert calls == ["Personal/Build Orchestrator/schedules/2026-w1.yaml"]


def test_inbound_move_into_schedules_from_non_bo_path_is_validated(monkeypatch, vault_dir):
    """B1: 'Moving an arbitrary file from a non-BO path into .../schedules/
    bypasses the guard' -- an inbound move must be validated as if it were a
    fresh write at the destination."""
    source = vault_dir / "Clients" / "arbitrary.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(VALID_SCHEDULE_NEW)

    monkeypatch.setattr(
        bo_contract, "preflight_schedule_rewrite",
        lambda schedule_path, new_builds, mode="strict_new", timeout=None: {"ok": True, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        bo_contract, "validate_graph",
        lambda nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None: {
            "ok": False,
            "errors": [{"code": "unknown_project", "message": "bad project"}],
            "warnings": [],
        },
    )
    result = bo_guard.evaluate_path_mutation(
        PathMutationContext(path="Clients/arbitrary.md", operation="move", destination=SCHEDULE_PATH)
    )
    assert any(i.rule_id == "bo-guard-schedule-graph" for i in result.issues)


def test_inbound_move_of_valid_content_has_no_issues(monkeypatch, vault_dir):
    source = vault_dir / "Clients" / "arbitrary.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(VALID_SCHEDULE_NEW)
    monkeypatch.setattr(
        bo_contract, "preflight_schedule_rewrite",
        lambda schedule_path, new_builds, mode="strict_new", timeout=None: {"ok": True, "errors": [], "warnings": []},
    )
    result = bo_guard.evaluate_path_mutation(
        PathMutationContext(path="Clients/arbitrary.md", operation="move", destination=SCHEDULE_PATH)
    )
    assert result.issues == []


def test_outbound_move_from_bo_path_to_non_bo_path_still_evaluates_source(monkeypatch):
    """Sanity: destination-side inbound checks must not replace source-side
    checks for an ordinary outbound move."""
    monkeypatch.setattr(
        bo_contract, "preflight_schedule_move",
        lambda schedule_path, timeout=None: {
            "ok": False,
            "errors": [{"code": "schedule_move_would_orphan_binding", "message": "bound", "build_id": "x"}],
            "warnings": [],
        },
    )
    result = bo_guard.evaluate_path_mutation(
        PathMutationContext(path=SCHEDULE_PATH, operation="move", destination="Clients/moved-out.yaml")
    )
    assert any(i.rule_id == "bo-guard-schedule-move" for i in result.issues)


# --------------------------------------------------------------------------
# B7 (codex-review-bo-authoring-contract-v2): a pending/ready spec rewrite
# must also revalidate the whole graph of any schedule it's already bound
# to, using the PROPOSED spec bytes.
# --------------------------------------------------------------------------


def test_spec_rewrite_not_bound_to_any_schedule_skips_graph_revalidation(monkeypatch):
    """Default fixture behaviour: a spec with no source_schedule has nothing
    for the B7 path to revalidate."""
    monkeypatch.setattr(
        bo_contract, "preflight_ids",
        lambda build_ids, timeout=None: {"results": {"scratch-build": "pending"}},
    )

    def boom(nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None):
        raise AssertionError("validate_graph should not be called when not bound to a schedule")

    monkeypatch.setattr(bo_contract, "validate_graph", boom)
    result = bo_guard.evaluate_content(
        WriteContext(path=SPEC_PATH, old_content="old body", new_content="new body", tool="vault_write")
    )
    assert result.issues == []


def test_spec_rewrite_bound_to_schedule_revalidates_whole_graph_with_proposed_bytes(monkeypatch, vault_dir):
    """B7 reproduction: an individually well-formed spec rewrite (e.g.
    changing its project) can still invalidate its referring schedule's
    project/dependency authority -- the schedule's complete graph must be
    revalidated using the PROPOSED spec bytes, not what's on disk."""
    schedule_path = SCHEDULE_PATH
    schedule_content = (
        "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# scratch\n\nbuilds:\n\n"
        "  - id: scratch-build\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/scratch-build.md\n"
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-scratch.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(schedule_content)

    monkeypatch.setattr(
        bo_contract, "preflight_ids",
        lambda build_ids, timeout=None: {"results": {"scratch-build": "pending"}},
    )
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"scratch-build": schedule_path}},
    )

    seen_calls = []

    def fake_validate_graph(nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None):
        seen_calls.append((nodes, new_ids, mode))
        # Baseline (compat_existing) reflects the schedule as it exists on
        # disk today -- clean, no mix. Only the PROPOSED graph (strict_new,
        # built_id's node substituted with the new bytes) introduces the
        # mixed_project_schedule finding, proving this specific edit is what
        # invalidates the graph (vault-bo-authoring-enforcement-readiness-v1,
        # Phase 1's baseline-vs-proposed comparison must not treat a
        # genuinely NEW condition as pre-existing/compatible).
        if mode == "compat_existing":
            return {"ok": True, "errors": [], "warnings": []}
        return {
            "ok": False,
            "errors": [{"code": "mixed_project_schedule", "message": "project changed", "build_id": "scratch-build"}],
            "warnings": [],
        }

    monkeypatch.setattr(bo_contract, "validate_graph", fake_validate_graph)

    result = bo_guard.evaluate_content(
        WriteContext(path=SPEC_PATH, old_content="old body", new_content="new spec body", tool="vault_write")
    )
    assert any(i.rule_id == "bo-guard-spec-rewrite-graph" for i in result.issues)
    assert seen_calls
    nodes, new_ids, mode = seen_calls[-1]
    assert mode == "strict_new"
    assert new_ids == ["scratch-build"]
    assert nodes[0]["build_id"] == "scratch-build"
    assert nodes[0]["spec_markdown"] == "new spec body"


def test_spec_rewrite_bound_to_schedule_with_valid_graph_has_no_issues(monkeypatch, vault_dir):
    schedule_path = SCHEDULE_PATH
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-scratch.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(VALID_SCHEDULE_OLD)

    monkeypatch.setattr(
        bo_contract, "preflight_ids",
        lambda build_ids, timeout=None: {"results": {"existing-build": "pending"}},
    )
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"existing-build": schedule_path}},
    )
    result = bo_guard.evaluate_content(
        WriteContext(
            path="Personal/Build Orchestrator/specs/existing-build.md",
            old_content="old body", new_content="new body", tool="vault_write",
        )
    )
    assert result.issues == []


# --------------------------------------------------------------------------
# B8 (codex-review-bo-authoring-contract-v2): destination validation must
# run independent of whether the source is also a BO path, and inbound
# directory moves must enumerate and validate every nested artifact.
# --------------------------------------------------------------------------


def test_specs_to_schedules_cross_move_validates_destination_even_when_source_is_bo(monkeypatch, vault_dir):
    """A BO-path source (specs/) must not exempt a BO-path destination
    (schedules/) from validation."""
    source = vault_dir / "Personal" / "Build Orchestrator" / "specs" / "scratch-build.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("---\nbuild_id: scratch-build\n---\n\nordinary spec body, not a schedule")

    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"scratch-build": None}})

    result = bo_guard.evaluate_path_mutation(
        PathMutationContext(path=SPEC_PATH, operation="move", destination=SCHEDULE_PATH)
    )
    assert any(i.rule_id == "bo-guard-schedule-rewrite" for i in result.issues)


def test_schedules_to_specs_cross_move_validates_destination_type(monkeypatch, vault_dir):
    source = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-w1.yaml"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(VALID_SCHEDULE_OLD)

    monkeypatch.setattr(
        bo_contract, "preflight_schedule_move",
        lambda schedule_path, timeout=None: {"ok": True, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"2026-w1": None}})
    monkeypatch.setattr(
        bo_contract, "preflight_spec_validate",
        lambda build_id, spec_markdown, spec_path, mode="compat_existing", timeout=None: {
            "ok": False,
            "errors": [{"code": "spec_missing_field", "message": "not a spec", "build_id": build_id}],
        },
    )
    result = bo_guard.evaluate_path_mutation(
        PathMutationContext(
            path="Personal/Build Orchestrator/schedules/2026-w1.yaml", operation="move",
            destination="Personal/Build Orchestrator/specs/2026-w1.md",
        )
    )
    assert any(i.rule_id == "bo-guard-spec-rewrite" for i in result.issues)


def test_inbound_directory_move_enumerates_and_validates_nested_files(monkeypatch, vault_dir):
    source_dir = vault_dir / "Clients" / "incoming"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "a.yaml").write_text(VALID_SCHEDULE_NEW)
    (source_dir / "b.yaml").write_text("not: valid: yaml: schedule: [")

    monkeypatch.setattr(
        bo_contract, "preflight_schedule_rewrite",
        lambda schedule_path, new_builds, mode="strict_new", timeout=None: {"ok": True, "errors": [], "warnings": []},
    )
    result = bo_guard.evaluate_path_mutation(
        PathMutationContext(
            path="Clients/incoming", operation="move",
            destination="Personal/Build Orchestrator/schedules/incoming",
        )
    )
    assert any(i.rule_id == "bo-guard-schedule-rewrite" for i in result.issues)


def test_inbound_directory_move_fails_closed_on_unevaluable_nested_file(vault_dir):
    source_dir = vault_dir / "Clients" / "incoming2"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "binary.yaml").write_bytes(b"\xff\xfe\x00\x01binary garbage")

    result = bo_guard.evaluate_path_mutation(
        PathMutationContext(
            path="Clients/incoming2", operation="move",
            destination="Personal/Build Orchestrator/schedules/incoming2",
        )
    )
    assert any(i.rule_id == "bo-guard-inbound-directory-move" for i in result.issues)


# --------------------------------------------------------------------------
# HI-1 (opus-review-bo-authoring-contract-v4): every guard evaluation must
# assert the adapter's schema/contract version before trusting any of its
# preflight/graph results, and a failure must never be silently converted
# into permission by a hardcoded fallback.
# --------------------------------------------------------------------------


def test_evaluate_content_rejects_on_adapter_version_drift(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no further adapter call should run once the version check fails")

    monkeypatch.setattr(
        bo_contract, "check_version",
        lambda timeout=None: (_ for _ in ()).throw(bo_contract.BOContractError("version_mismatch", "drift")),
    )
    monkeypatch.setattr(bo_contract, "preflight_schedule_rewrite", boom)
    monkeypatch.setattr(bo_contract, "validate_graph", boom)

    result = bo_guard.evaluate_content(
        WriteContext(path=SCHEDULE_PATH, old_content=VALID_SCHEDULE_OLD, new_content=VALID_SCHEDULE_NEW, tool="vault_write")
    )
    assert len(result.issues) == 1
    assert result.issues[0].rule_id == "bo-guard-adapter-version"
    assert result.issues[0].severity == "reject"

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    with pytest.raises(bo_guard.BOGuardError):
        bo_guard.enforce(
            WriteContext(path=SCHEDULE_PATH, old_content=VALID_SCHEDULE_OLD, new_content=VALID_SCHEDULE_NEW, tool="vault_write")
        )


def test_evaluate_path_mutation_rejects_on_adapter_version_drift(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no further adapter call should run once the version check fails")

    monkeypatch.setattr(
        bo_contract, "check_version",
        lambda timeout=None: (_ for _ in ()).throw(bo_contract.BOContractError("version_mismatch", "drift")),
    )
    monkeypatch.setattr(bo_contract, "preflight_schedule_move", boom)
    monkeypatch.setattr(bo_contract, "preflight_ids", boom)

    result = bo_guard.evaluate_path_mutation(PathMutationContext(path=SCHEDULE_PATH, operation="delete"))
    assert len(result.issues) == 1
    assert result.issues[0].rule_id == "bo-guard-adapter-version"
    assert result.issues[0].severity == "reject"

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    with pytest.raises(bo_guard.BOGuardError):
        bo_guard.enforce_path_mutation(PathMutationContext(path=SCHEDULE_PATH, operation="delete"))


def test_adapter_unavailable_for_version_check_is_also_reject(monkeypatch):
    monkeypatch.setattr(
        bo_contract, "check_version",
        lambda timeout=None: (_ for _ in ()).throw(bo_contract.BOContractError("adapter_missing", "not found")),
    )
    result = bo_guard.evaluate_content(
        WriteContext(path=SPEC_PATH, old_content="old body", new_content="new body", tool="vault_write")
    )
    assert result.issues[0].rule_id == "bo-guard-adapter-version"


def test_freely_mutable_statuses_propagates_adapter_failure_with_no_fallback(monkeypatch):
    """HI-1: the old fallback silently substituted a hardcoded status set on
    ANY adapter failure. There is no fallback now -- a failure propagates as
    BOContractError for the caller to turn into a reject issue."""
    monkeypatch.setattr(bo_guard, "_freely_mutable_statuses", _REAL_FREELY_MUTABLE_STATUSES)
    monkeypatch.setattr(bo_guard, "_freely_mutable_statuses_cache", None)
    monkeypatch.setattr(
        bo_contract, "freely_mutable_statuses",
        lambda timeout=None: (_ for _ in ()).throw(bo_contract.BOContractError("adapter_missing", "gone")),
    )
    with pytest.raises(bo_contract.BOContractError):
        bo_guard._freely_mutable_statuses()


def test_spec_rewrite_status_check_failure_is_reject_not_silent_allow(monkeypatch):
    monkeypatch.setattr(bo_guard, "_freely_mutable_statuses", _REAL_FREELY_MUTABLE_STATUSES)
    monkeypatch.setattr(bo_guard, "_freely_mutable_statuses_cache", None)
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"scratch-build": "proposed"}})
    monkeypatch.setattr(
        bo_contract, "freely_mutable_statuses",
        lambda timeout=None: (_ for _ in ()).throw(bo_contract.BOContractError("adapter_missing", "gone")),
    )
    result = bo_guard.evaluate_content(
        WriteContext(path=SPEC_PATH, old_content="old body", new_content="new body", tool="vault_write")
    )
    assert any(i.rule_id == "bo-guard-adapter-unavailable" and i.severity == "reject" for i in result.issues)


def test_spec_move_status_check_failure_is_reject_not_silent_allow(monkeypatch):
    monkeypatch.setattr(bo_guard, "_freely_mutable_statuses", _REAL_FREELY_MUTABLE_STATUSES)
    monkeypatch.setattr(bo_guard, "_freely_mutable_statuses_cache", None)
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"scratch-build": "proposed"}})
    monkeypatch.setattr(
        bo_contract, "freely_mutable_statuses",
        lambda timeout=None: (_ for _ in ()).throw(bo_contract.BOContractError("adapter_missing", "gone")),
    )
    result = bo_guard.evaluate_path_mutation(PathMutationContext(path=SPEC_PATH, operation="delete"))
    assert any(i.rule_id == "bo-guard-adapter-unavailable" and i.severity == "reject" for i in result.issues)


# --------------------------------------------------------------------------
# HI-2 (opus-review-bo-authoring-contract-v4): a spec rewrite's schedule-
# graph revalidation must fail closed (not return []) when a known binding
# can't be resolved, and must also discover referring schedules from disk
# for a build not yet ingested into the DB.
# --------------------------------------------------------------------------


def test_spec_rewrite_discovers_referring_schedule_from_disk_when_not_in_db(monkeypatch, vault_dir):
    """HI-2b: an un-ingested build (no DB row) can still be declared in an
    on-disk schedule file. The DB lookup alone (preflight_source_schedule)
    misses this; disk discovery must catch it."""
    schedule_content = (
        "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# scratch\n\nbuilds:\n\n"
        "  - id: scratch-build\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/scratch-build.md\n"
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-scratch.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(schedule_content)

    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"scratch-build": None}})
    # DB has no binding at all -- this is the un-ingested case.
    monkeypatch.setattr(bo_contract, "preflight_source_schedule", lambda build_ids, timeout=None: {"results": {"scratch-build": None}})

    seen_calls = []

    def fake_validate_graph(nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None):
        seen_calls.append((nodes, new_ids))
        # duplicate_id_in_graph, not mixed_project_schedule: this test is
        # about disk-based schedule discovery (HI-2b), not mixed-project
        # baseline-vs-proposed semantics (vault-bo-authoring-enforcement-
        # readiness-v1, Phase 1) -- any reject-severity graph code proves the
        # same thing here.
        return {"ok": False, "errors": [{"code": "duplicate_id_in_graph", "message": "bad", "build_id": "scratch-build"}], "warnings": []}

    monkeypatch.setattr(bo_contract, "validate_graph", fake_validate_graph)

    result = bo_guard.evaluate_content(
        WriteContext(path=SPEC_PATH, old_content="old body", new_content="new spec body", tool="vault_write")
    )
    assert any(i.rule_id == "bo-guard-spec-rewrite-graph" for i in result.issues)
    assert seen_calls
    nodes, new_ids = seen_calls[-1]
    assert new_ids == ["scratch-build"]
    assert nodes[0]["spec_markdown"] == "new spec body"


def test_spec_rewrite_bound_schedule_missing_on_disk_fails_closed(monkeypatch):
    """HI-2: previously returned [] (silently skipped revalidation) when the
    DB-bound schedule file doesn't exist on disk. Must reject instead."""
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"scratch-build": "pending"}})
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"scratch-build": "Personal/Build Orchestrator/schedules/does-not-exist.yaml"}},
    )
    result = bo_guard.evaluate_content(
        WriteContext(path=SPEC_PATH, old_content="old body", new_content="new body", tool="vault_write")
    )
    assert any(i.rule_id == "bo-guard-spec-rewrite-graph" and i.severity == "reject" for i in result.issues)


def test_spec_rewrite_bound_schedule_unparseable_fails_closed(monkeypatch, vault_dir):
    """HI-2: previously returned [] when the DB-bound schedule couldn't be
    parsed. Must reject instead."""
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-scratch.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text("not: [valid, yaml:::")

    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"scratch-build": "pending"}})
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"scratch-build": SCHEDULE_PATH}},
    )
    result = bo_guard.evaluate_content(
        WriteContext(path=SPEC_PATH, old_content="old body", new_content="new body", tool="vault_write")
    )
    assert any(i.rule_id == "bo-guard-spec-rewrite-graph" and i.severity == "reject" for i in result.issues)


def test_spec_rewrite_bound_schedule_missing_entry_fails_closed(monkeypatch, vault_dir):
    """HI-2: the DB says this build is bound to a schedule, but that
    schedule's builds: list doesn't actually contain the build's entry --
    previously the graph validated vacuously (without the proposed bytes);
    must reject instead."""
    schedule_content = (
        "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# scratch\n\nbuilds:\n\n"
        "  - id: some-other-build\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/some-other-build.md\n"
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-scratch.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(schedule_content)

    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"scratch-build": "pending"}})
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"scratch-build": SCHEDULE_PATH}},
    )

    def boom(*a, **k):
        raise AssertionError("validate_graph should not be called when the build's entry is absent from its bound schedule")

    monkeypatch.setattr(bo_contract, "validate_graph", boom)
    result = bo_guard.evaluate_content(
        WriteContext(path=SPEC_PATH, old_content="old body", new_content="new body", tool="vault_write")
    )
    assert any(i.rule_id == "bo-guard-spec-rewrite-graph" and i.severity == "reject" for i in result.issues)


# --------------------------------------------------------------------------
# MD-1 (opus-review-bo-authoring-contract-v4): an undecodable/binary inbound
# move source must fail closed, not silently skip validation.
# --------------------------------------------------------------------------


def test_inbound_move_of_binary_file_fails_closed(vault_dir):
    source = vault_dir / "Clients" / "blob.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"\xff\xfe\x00\x01binary garbage, not a directory")

    result = bo_guard.evaluate_path_mutation(
        PathMutationContext(
            path="Clients/blob.bin", operation="move",
            destination="Personal/Build Orchestrator/schedules/blob.yaml",
        )
    )
    assert any(i.rule_id == "bo-guard-inbound-move-undecodable" and i.severity == "reject" for i in result.issues)


# --------------------------------------------------------------------------
# LO-1 (opus-review-bo-authoring-contract-v4): moving/deleting a
# SUBDIRECTORY of specs/ or schedules/ must enumerate the files nested
# inside it, not look up the directory itself as one exact authority path.
# --------------------------------------------------------------------------


def test_moving_subdirectory_of_schedules_enumerates_nested_files(monkeypatch, vault_dir):
    subdir = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "wk34"
    subdir.mkdir(parents=True)
    (subdir / "2026-w34.yaml").write_text(VALID_SCHEDULE_OLD)

    seen_paths = []

    def fake_preflight_schedule_move(schedule_path, timeout=None):
        seen_paths.append(schedule_path)
        return {"ok": False, "errors": [{"code": "schedule_move_would_remove_bound_authority", "message": "x", "build_id": "existing-build"}], "warnings": []}

    monkeypatch.setattr(bo_contract, "preflight_schedule_move", fake_preflight_schedule_move)
    result = bo_guard.evaluate_path_mutation(
        PathMutationContext(
            path="Personal/Build Orchestrator/schedules/wk34", operation="move",
            destination="Personal/Build Orchestrator/schedules/wk34-renamed",
        )
    )
    assert seen_paths == ["Personal/Build Orchestrator/schedules/wk34/2026-w34.yaml"]
    assert any(i.rule_id == "bo-guard-schedule-move" for i in result.issues)
