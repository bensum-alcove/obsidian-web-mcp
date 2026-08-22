"""End-to-end regression tests for BL-1 and required review item 9
(opus-review-bo-authoring-contract-v4), exercised through the REAL
vault.write_file_atomic / vault.move_path / vault.delete_path -- not
bo_guard's functions called directly -- because BL-1's defect lived in
*vault.py* (constructing WriteContext/PathMutationContext from the caller's
raw path string) rather than in bo_guard.py itself. bo_contract is
monkeypatched (never touches the real adapter subprocess); the vault
filesystem is real (via the `vault_dir` fixture) so resolve_vault_path's own
symlink/`.`/`..`/`//` normalisation genuinely runs.
"""

import os

import pytest

from obsidian_vault_mcp import bo_contract, bo_guard, config, vault

SCHEDULE_REL = "Personal/Build Orchestrator/schedules/x.yaml"
SPEC_REL = "Personal/Build Orchestrator/specs/scratch-build.md"

VALID_SCHEDULE = (
    "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# scratch\n\nbuilds:\n\n"
    "  - id: existing-build\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
    "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/existing-build.md\n"
)


@pytest.fixture(autouse=True)
def _enforce_bo_guard_only(monkeypatch):
    """Isolate these tests to the BO guard: write_contract stays off so its
    own unrelated rules can't interfere with a pure BL-1/item-9 regression."""
    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE", "off")
    monkeypatch.setenv("VAULT_OPTIMISTIC_CONCURRENCY_MODE", "off")
    monkeypatch.setattr(
        bo_contract, "check_version",
        lambda timeout=None: {
            "ok": True,
            "schema_version": bo_contract.CONSUMER_SCHEMA_VERSION,
            "contract_version": bo_contract.EXPECTED_CONTRACT_VERSION,
            "known_statuses": ["proposed", "pending", "ready", "dispatched", "done"],
            "terminal_statuses": ["done"],
            "dispatched_statuses": ["dispatched"],
        },
    )
    monkeypatch.setattr(bo_guard, "_freely_mutable_statuses_cache", None)


def _reject_everything(monkeypatch):
    """Make any schedule-graph validation reject, and any spec-status/schedule-move
    preflight reject -- so if (and only if) the guard is actually evaluating a
    given path, the operation must be blocked."""
    monkeypatch.setattr(
        bo_contract, "preflight_schedule_rewrite",
        lambda schedule_path, new_builds, mode="strict_new", timeout=None: {"ok": True, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        bo_contract, "validate_graph",
        lambda nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None: {
            "ok": False,
            "errors": [{"code": "unknown_project", "message": "rejected by test"}],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        bo_contract, "preflight_schedule_move",
        lambda schedule_path, timeout=None: {
            "ok": False,
            "errors": [{"code": "schedule_move_would_remove_bound_authority", "message": "bound", "build_id": "existing-build"}],
            "warnings": [],
        },
    )
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {bid: "dispatched" for bid in build_ids}})


ALIAS_FORMS = [
    "Personal/Build Orchestrator//schedules/x.yaml",
    "./Personal/Build Orchestrator/schedules/x.yaml",
    "Personal/Build Orchestrator/./schedules/x.yaml",
]


@pytest.mark.parametrize("alias_path", ALIAS_FORMS)
def test_write_through_path_alias_is_still_guarded(monkeypatch, vault_dir, alias_path):
    """BL-1: these aliases resolve to the exact same file as SCHEDULE_REL but
    used to skip _is_bo_path's literal-prefix check entirely."""
    _reject_everything(monkeypatch)
    with pytest.raises(bo_guard.BOGuardError):
        vault.write_file_atomic(alias_path, VALID_SCHEDULE, tool="vault_write")
    assert not (vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "x.yaml").exists()


def test_write_through_symlinked_directory_is_still_guarded(monkeypatch, vault_dir):
    """BL-1's fourth witness: a vault-internal symlink pointing at the real
    schedules/ directory must resolve to the canonical path before the guard
    evaluates it."""
    _reject_everything(monkeypatch)
    real_dir = vault_dir / "Personal" / "Build Orchestrator" / "schedules"
    real_dir.mkdir(parents=True)
    alias_dir = vault_dir / "Personal" / "alias-schedules"
    os.symlink(real_dir, alias_dir)

    with pytest.raises(bo_guard.BOGuardError):
        vault.write_file_atomic("Personal/alias-schedules/x.yaml", VALID_SCHEDULE, tool="vault_write")
    assert not (real_dir / "x.yaml").exists()


@pytest.mark.parametrize("alias_source", [
    "Personal/Build Orchestrator//schedules/bound.yaml",
    "./Personal/Build Orchestrator/schedules/bound.yaml",
])
def test_move_source_through_path_alias_is_still_guarded(monkeypatch, vault_dir, alias_source):
    """BL-1: an aliased MOVE SOURCE must not bypass source-side evaluation
    (e.g. moving a schedule still bound to a build)."""
    _reject_everything(monkeypatch)
    real_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "bound.yaml"
    real_file.parent.mkdir(parents=True, exist_ok=True)
    real_file.write_text(VALID_SCHEDULE)

    with pytest.raises(bo_guard.BOGuardError):
        vault.move_path(alias_source, "Clients/moved-out.yaml")
    assert real_file.exists()
    assert not (vault_dir / "Clients" / "moved-out.yaml").exists()


@pytest.mark.parametrize("alias_destination", [
    "Personal/Build Orchestrator//schedules/inbound.yaml",
    "./Personal/Build Orchestrator/schedules/inbound.yaml",
])
def test_move_destination_through_path_alias_is_still_guarded(monkeypatch, vault_dir, alias_destination):
    """BL-1: an aliased MOVE DESTINATION must not bypass inbound-move
    validation (B8's destination-side check)."""
    _reject_everything(monkeypatch)
    source = vault_dir / "Clients" / "arbitrary.yaml"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(VALID_SCHEDULE)

    with pytest.raises(bo_guard.BOGuardError):
        vault.move_path("Clients/arbitrary.yaml", alias_destination)
    assert source.exists()
    assert not (vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "inbound.yaml").exists()


@pytest.mark.parametrize("alias_path", [
    "Personal/Build Orchestrator//schedules/bound.yaml",
    "./Personal/Build Orchestrator/schedules/bound.yaml",
])
def test_delete_through_path_alias_is_still_guarded(monkeypatch, vault_dir, alias_path):
    """BL-1: an aliased DELETE target must not bypass the schedule-move
    (delete-is-a-move-to-.trash) preflight."""
    _reject_everything(monkeypatch)
    real_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "bound.yaml"
    real_file.parent.mkdir(parents=True, exist_ok=True)
    real_file.write_text(VALID_SCHEDULE)

    with pytest.raises(bo_guard.BOGuardError):
        vault.delete_path(alias_path)
    assert real_file.exists()


def test_canonical_path_control_is_also_guarded(monkeypatch, vault_dir):
    """Sanity control: the exact canonical (non-aliased) path must behave
    identically to the alias witnesses above -- proves the aliases aren't
    passing for some unrelated reason."""
    _reject_everything(monkeypatch)
    with pytest.raises(bo_guard.BOGuardError):
        vault.write_file_atomic(SCHEDULE_REL, VALID_SCHEDULE, tool="vault_write")
    assert not (vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "x.yaml").exists()


# --------------------------------------------------------------------------
# Required review item 9: bo_guard re-evaluates at the activation boundary
# (immediately before the filesystem commit), narrowing the window in which
# a change to a referring schedule's bytes between the first check and the
# commit could otherwise let a now-invalid write through.
# --------------------------------------------------------------------------


def test_second_activation_boundary_check_catches_a_change_and_blocks_commit(monkeypatch, vault_dir):
    """Deterministic proxy for the real race: validate_graph passes on its
    FIRST invocation (as if the schedule were still consistent) and fails on
    its SECOND (as if a concurrent writer had changed the referring schedule
    in between) -- proving write_file_atomic really does re-run the guard a
    second time immediately before committing, and that a late failure still
    blocks the write."""
    schedule_content = (
        "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# scratch\n\nbuilds:\n\n"
        "  - id: scratch-build\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/scratch-build.md\n"
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-scratch.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(schedule_content)

    spec_file = vault_dir / "Personal" / "Build Orchestrator" / "specs" / "scratch-build.md"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    spec_file.write_text("old body")

    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"scratch-build": "pending"}})
    monkeypatch.setattr(
        bo_contract, "preflight_spec_validate",
        lambda build_id, spec_markdown, spec_path, mode="compat_existing", timeout=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"scratch-build": "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml"}},
    )

    calls = {"n": 0}

    def fake_validate_graph(nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None):
        # Only count the PROPOSED (strict_new) call in each guard evaluation
        # -- Phase 1 (vault-bo-authoring-enforcement-readiness-v1) added an
        # extra baseline (compat_existing) call per evaluation purely to
        # detect pre-existing mixed_project_schedule conditions; it carries
        # no enforcement decision of its own and must always read as clean
        # here so it never confuses this test's "first vs. second real
        # evaluation" counter.
        if mode == "compat_existing":
            return {"ok": True, "errors": [], "warnings": []}
        calls["n"] += 1
        if calls["n"] == 1:
            return {"ok": True, "errors": [], "warnings": []}
        return {"ok": False, "errors": [{"code": "mixed_project_schedule", "message": "changed mid-write", "build_id": "scratch-build"}], "warnings": []}

    monkeypatch.setattr(bo_contract, "validate_graph", fake_validate_graph)

    with pytest.raises(bo_guard.BOGuardError):
        vault.write_file_atomic(SPEC_REL, "new body", tool="vault_write")

    assert calls["n"] == 2, "expected exactly two guard evaluations: the early one and the activation-boundary re-check"
    assert spec_file.read_text() == "old body", "the second check must have prevented the commit"


def test_activation_boundary_check_passes_through_when_nothing_changed(monkeypatch, vault_dir):
    """Sanity control: when both guard evaluations agree the write is valid,
    the commit proceeds normally (the re-check is not a spurious blocker)."""
    schedule_content = (
        "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# scratch\n\nbuilds:\n\n"
        "  - id: scratch-build\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/scratch-build.md\n"
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-scratch.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(schedule_content)

    spec_file = vault_dir / "Personal" / "Build Orchestrator" / "specs" / "scratch-build.md"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    spec_file.write_text("old body")

    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"scratch-build": "pending"}})
    monkeypatch.setattr(
        bo_contract, "preflight_spec_validate",
        lambda build_id, spec_markdown, spec_path, mode="compat_existing", timeout=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"scratch-build": "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml"}},
    )

    calls = {"n": 0}

    def fake_validate_graph(nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None):
        # See the sibling test above: only strict_new (proposed) calls are
        # counted, since Phase 1's added compat_existing baseline call
        # carries no enforcement decision of its own.
        if mode == "compat_existing":
            return {"ok": True, "errors": [], "warnings": []}
        calls["n"] += 1
        return {"ok": True, "errors": [], "warnings": []}

    monkeypatch.setattr(bo_contract, "validate_graph", fake_validate_graph)

    is_new, size = vault.write_file_atomic(SPEC_REL, "new body", tool="vault_write")
    assert not is_new
    assert calls["n"] == 2
    assert spec_file.read_text() == "new body"
