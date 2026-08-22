"""Regression tests for vault-bo-authoring-guard-remediation-v6
(opus-review-bo-authoring-contract-v5's BLOCKER/HIGH/MEDIUM/LOW findings).

bo_contract's adapter calls are monkeypatched throughout (never touch the
real subprocess/DB) except where a test explicitly targets the real
corpus/adapter and is guarded with a skipif, matching this repo's existing
convention (see test_bo_contract.py's _ADAPTER_PRESENT tests and
test_bo_guard_vault_integration.py's real-filesystem-via-vault_dir style).
"""

import os

import pytest

from obsidian_vault_mcp import bo_contract, bo_guard, config, vault
from obsidian_vault_mcp.write_contract import PathMutationContext, WriteContext

REAL_CORPUS_DIR = "/home/ben_sum/vaults/bs-brain/Personal/Build Orchestrator/schedules"

# The exact real-corpus shape that broke the old local parser: a `>`
# blockquote program-note line right after the heading, which parses as a
# YAML block-scalar indicator under a naive whole-body yaml.safe_load.
REAL_SHAPED_SCHEDULE = (
    "---\ntype: schedule\nproject: edge-trading-system\n---\n\n"
    "# Week 2026-W99 -- Test\n\n"
    "> Contradiction lint from the hot-cache/pre-flight initiative. Read-only audit, no code changes.\n\n"
    "builds:\n\n"
    "  - id: race-shaped\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
    "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/race-shaped.md\n"
)


def _old_naive_parser(content: str) -> list | None:
    """The exact removed bo_guard.parse_schedule_builds implementation,
    reproduced here only to prove it fails on the real-shaped fixture above
    -- it must never be reintroduced as a competing parser."""
    import frontmatter
    import yaml

    try:
        post = frontmatter.loads(content)
        body = yaml.safe_load(post.content) or {}
    except Exception:
        return None
    builds = body.get("builds") if isinstance(body, dict) else None
    return builds if isinstance(builds, list) else None


@pytest.fixture(autouse=True)
def _stub_check_version_ok(monkeypatch):
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


# --------------------------------------------------------------------------
# BL5-1: exactly one schedule parser, and it handles the real corpus shape.
# --------------------------------------------------------------------------


def test_old_naive_parser_fails_on_real_shaped_blockquote():
    """Sanity control proving the fixture is a faithful reproduction of the
    real bug: the removed local parser chokes on it."""
    assert _old_naive_parser(REAL_SHAPED_SCHEDULE) is None


def test_canonical_parser_handles_real_shaped_blockquote():
    builds = bo_guard.schedule_builds_from_content(REAL_SHAPED_SCHEDULE, source_name="test")
    assert builds is not None
    assert [b["id"] for b in builds] == ["race-shaped"]


def test_canonical_parser_returns_none_on_genuinely_malformed_content():
    assert bo_guard.schedule_builds_from_content("not: [valid, yaml, {", source_name="test") is None


def test_schedule_rewrite_accepts_real_shaped_content_the_old_parser_rejected(monkeypatch, vault_dir):
    """End-to-end: a schedule rewrite in the real-shaped form used to be
    rejected as 'does not parse as a valid schedule builds: list' purely
    because of the old parser's blockquote bug, not any real problem with
    the content."""
    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    monkeypatch.setattr(
        bo_contract, "preflight_schedule_rewrite",
        lambda schedule_path, new_builds, mode="strict_new", timeout=None: {"ok": True, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        bo_contract, "validate_graph",
        lambda nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None:
            {"ok": True, "errors": [], "warnings": []},
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-real-shaped.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)

    is_new, _ = vault.write_file_atomic(
        "Personal/Build Orchestrator/schedules/2026-W99-real-shaped.yaml",
        REAL_SHAPED_SCHEDULE, tool="vault_write",
    )
    assert is_new
    assert schedule_file.read_text() == REAL_SHAPED_SCHEDULE


@pytest.mark.skipif(not os.path.isdir(REAL_CORPUS_DIR), reason="real BS Brain schedule corpus not present on this host")
def test_canonical_parser_parses_the_entire_real_schedule_corpus():
    """MD5-2/BL5-1 evidence: every real, currently-deployed schedule file
    must parse via the canonical parser. This is the exact check that found
    the old parser failed on 54-56% of this corpus."""
    import pathlib

    corpus = sorted(pathlib.Path(REAL_CORPUS_DIR).glob("*.yaml"))
    assert len(corpus) > 50, "sanity: expected the real corpus to be non-trivially sized"
    failures = []
    for f in corpus:
        content = f.read_text(encoding="utf-8")
        if bo_guard.schedule_builds_from_content(content, source_name=str(f)) is None:
            failures.append(f.name)
    assert failures == [], f"{len(failures)}/{len(corpus)} real schedules failed to parse: {failures[:10]}"


def test_parser_module_load_failure_is_adapter_unavailable_not_content_failure(monkeypatch):
    """The two failure modes must stay distinguishable: a broken/missing
    contract.py module is a systemic BOContractError, never silently
    conflated with 'this one document happens to be malformed'."""
    monkeypatch.setattr(bo_contract, "_schedule_parser_module", None)
    monkeypatch.setattr(config, "BO_AUTHORING_CONTRACT_PATH", "/nonexistent/path/authoring_contract.py")
    with pytest.raises(bo_contract.BOContractError):
        bo_guard.schedule_builds_from_content(REAL_SHAPED_SCHEDULE, source_name="test")


# --------------------------------------------------------------------------
# HI5-1: move_path/delete_path activation-boundary re-check.
# --------------------------------------------------------------------------


def test_move_path_reevaluates_guard_immediately_before_the_mutation(monkeypatch, vault_dir):
    """Deterministic proxy for the real race (same methodology as this
    repo's existing write_file_atomic activation-boundary test): the FIRST
    evaluate_path_mutation call passes; the SECOND -- simulating a
    concurrent writer that landed malformed bytes on a nested file between
    enumeration and the filesystem move -- fails. move_path must call
    enforce_path_mutation twice and the second failure must block the move.

    A real two-process wall-clock race (barrier-synchronised mover +
    mutator, swept offsets) was used to develop and verify this fix
    directly: 7/12 offsets reproduced the bypass against the pre-fix code
    (single check) and 0/12 reproduced it against the fixed code (this
    test's assertions), at the identical offsets.
    """
    source = vault_dir / "Scratch" / "incoming"
    source.mkdir(parents=True)
    (source / "s00.yaml").write_text("placeholder")
    dest_parent = vault_dir / "Personal" / "Build Orchestrator" / "schedules"
    dest_parent.mkdir(parents=True)

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")

    calls = {"n": 0}

    def fake_evaluate(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            return bo_guard.BOGateResult(mode="enforce", issues=[])
        return bo_guard.BOGateResult(mode="enforce", issues=[
            bo_guard.ValidationIssue("bo-guard-inbound-directory-move", "reject", "changed mid-move"),
        ])

    monkeypatch.setattr(bo_guard, "evaluate_path_mutation", fake_evaluate)

    with pytest.raises(bo_guard.BOGuardError):
        vault.move_path("Scratch/incoming", "Personal/Build Orchestrator/schedules/incoming")

    assert calls["n"] == 2, "expected exactly two guard evaluations: the early one and the activation-boundary re-check"
    assert source.exists(), "the second check must have prevented the move from committing"
    assert not (dest_parent / "incoming").exists()


def test_delete_path_reevaluates_guard_immediately_before_the_mutation(monkeypatch, vault_dir):
    spec_file = vault_dir / "Personal" / "Build Orchestrator" / "specs" / "scratch.md"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    spec_file.write_text("body")

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")

    calls = {"n": 0}

    def fake_evaluate(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            return bo_guard.BOGateResult(mode="enforce", issues=[])
        return bo_guard.BOGateResult(mode="enforce", issues=[
            bo_guard.ValidationIssue("bo-guard-spec-move", "reject", "changed mid-delete"),
        ])

    monkeypatch.setattr(bo_guard, "evaluate_path_mutation", fake_evaluate)

    with pytest.raises(bo_guard.BOGuardError):
        vault.delete_path("Personal/Build Orchestrator/specs/scratch.md")

    assert calls["n"] == 2
    assert spec_file.exists(), "the second check must have prevented the delete from committing"


def test_move_path_commits_when_both_checks_agree(monkeypatch, vault_dir):
    """Sanity control: the added re-check is not a spurious blocker."""
    source = vault_dir / "Scratch" / "incoming"
    source.mkdir(parents=True)
    (source / "s00.yaml").write_text("placeholder")
    (vault_dir / "Personal" / "Build Orchestrator" / "schedules").mkdir(parents=True)

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    calls = {"n": 0}

    def fake_evaluate(ctx):
        calls["n"] += 1
        return bo_guard.BOGateResult(mode="enforce", issues=[])

    monkeypatch.setattr(bo_guard, "evaluate_path_mutation", fake_evaluate)

    assert vault.move_path("Scratch/incoming", "Personal/Build Orchestrator/schedules/incoming") is True
    assert calls["n"] == 2
    assert not source.exists()
    assert (vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "incoming" / "s00.yaml").exists()


# --------------------------------------------------------------------------
# HI5-3: a terminal/dispatched/duplicate-authority SIBLING must never block
# a benign edit, but the SAME finding attributed to the build actually being
# rewritten still must.
# --------------------------------------------------------------------------


def test_terminal_sibling_does_not_block_benign_spec_edit(monkeypatch, vault_dir):
    schedule_content = (
        "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# w\n\nbuilds:\n\n"
        "  - id: closed-sibling\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/closed-sibling.md\n"
        "  - id: rv6-target\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/rv6-target.md\n"
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-sib.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(schedule_content)

    spec_file = vault_dir / "Personal" / "Build Orchestrator" / "specs" / "rv6-target.md"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    spec_file.write_text("old body")

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"rv6-target": "pending"}})
    monkeypatch.setattr(
        bo_contract, "preflight_spec_validate",
        lambda build_id, spec_markdown, spec_path, mode="compat_existing", timeout=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"rv6-target": "Personal/Build Orchestrator/schedules/2026-W99-sib.yaml"}},
    )
    # Real adapter semantics being simulated: the sibling ("closed-sibling")
    # is already terminal, so validate_build_graph would raise
    # terminal_id_reuse attributed to IT, not to rv6-target.
    monkeypatch.setattr(
        bo_contract, "validate_graph",
        lambda nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None: {
            "ok": False,
            "errors": [{"code": "terminal_id_reuse", "message": "closed-sibling already reached terminal status", "build_id": "closed-sibling"}],
            "warnings": [],
        },
    )

    is_new, _ = vault.write_file_atomic(
        "Personal/Build Orchestrator/specs/rv6-target.md", "new body", tool="vault_write",
    )
    assert not is_new
    assert spec_file.read_text() == "new body"


def test_terminal_status_on_the_build_itself_still_blocks(monkeypatch, vault_dir):
    """Control: HI5-3's filter must be scoped to SIBLINGS only -- a
    terminal_id_reuse attributed to the build actually being rewritten must
    still reject."""
    schedule_content = (
        "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# w\n\nbuilds:\n\n"
        "  - id: rv6-target\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/rv6-target.md\n"
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-self.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(schedule_content)

    spec_file = vault_dir / "Personal" / "Build Orchestrator" / "specs" / "rv6-target.md"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    spec_file.write_text("old body")

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"rv6-target": "pending"}})
    monkeypatch.setattr(
        bo_contract, "preflight_spec_validate",
        lambda build_id, spec_markdown, spec_path, mode="compat_existing", timeout=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"rv6-target": "Personal/Build Orchestrator/schedules/2026-W99-self.yaml"}},
    )
    monkeypatch.setattr(
        bo_contract, "validate_graph",
        lambda nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None: {
            "ok": False,
            "errors": [{"code": "terminal_id_reuse", "message": "rv6-target already reached terminal status", "build_id": "rv6-target"}],
            "warnings": [],
        },
    )

    with pytest.raises(bo_guard.BOGuardError):
        vault.write_file_atomic("Personal/Build Orchestrator/specs/rv6-target.md", "new body", tool="vault_write")
    assert spec_file.read_text() == "old body"


def test_mixed_project_from_sibling_still_blocks(monkeypatch, vault_dir):
    """Control: a genuinely invalid graph (e.g. mixed-project) must still be
    rejected even though it's attributed to a sibling -- HI5-3's filter is
    scoped to _SIBLING_CONTEXT_ONLY_CODES, not a blanket sibling exemption."""
    schedule_content = (
        "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# w\n\nbuilds:\n\n"
        "  - id: other-project-sibling\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/other-project-sibling.md\n"
        "  - id: rv6-target\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/rv6-target.md\n"
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-mixed.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(schedule_content)

    spec_file = vault_dir / "Personal" / "Build Orchestrator" / "specs" / "rv6-target.md"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    spec_file.write_text("old body")

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"rv6-target": "pending"}})
    monkeypatch.setattr(
        bo_contract, "preflight_spec_validate",
        lambda build_id, spec_markdown, spec_path, mode="compat_existing", timeout=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"rv6-target": "Personal/Build Orchestrator/schedules/2026-W99-mixed.yaml"}},
    )
    def _validate_graph(nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None):
        # Baseline (compat_existing) is clean -- this edit is what
        # introduces the mix, not a pre-existing legacy condition
        # (vault-bo-authoring-enforcement-readiness-v1, Phase 1: only a
        # BASELINE-IDENTICAL mixed_project_schedule message is compatible;
        # a newly-introduced one must still reject).
        if mode == "compat_existing":
            return {"ok": True, "errors": [], "warnings": []}
        return {
            "ok": False,
            "errors": [{"code": "mixed_project_schedule", "message": "mixed projects", "path": "Personal/Build Orchestrator/schedules/2026-W99-mixed.yaml"}],
            "warnings": [],
        }

    monkeypatch.setattr(bo_contract, "validate_graph", _validate_graph)

    with pytest.raises(bo_guard.BOGuardError):
        vault.write_file_atomic("Personal/Build Orchestrator/specs/rv6-target.md", "new body", tool="vault_write")
    assert spec_file.read_text() == "old body"


def test_unchanged_baseline_mixed_project_does_not_block_benign_edit(monkeypatch, vault_dir):
    """vault-bo-authoring-enforcement-readiness-v1, Phase 1: a
    mixed_project_schedule finding whose message is byte-identical between
    the baseline (unedited) graph and the proposed graph describes a
    pre-existing legacy condition this content-only edit does not touch --
    it must not, by itself, make an otherwise-valid edit impossible."""
    schedule_content = (
        "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# w\n\nbuilds:\n\n"
        "  - id: other-project-sibling\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/other-project-sibling.md\n"
        "  - id: rv6-target\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/rv6-target.md\n"
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-legacy-mixed.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(schedule_content)

    spec_file = vault_dir / "Personal" / "Build Orchestrator" / "specs" / "rv6-target.md"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    spec_file.write_text("old body")

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"rv6-target": "pending"}})
    monkeypatch.setattr(
        bo_contract, "preflight_spec_validate",
        lambda build_id, spec_markdown, spec_path, mode="compat_existing", timeout=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"rv6-target": "Personal/Build Orchestrator/schedules/2026-W99-legacy-mixed.yaml"}},
    )

    same_message = (
        "schedule 'Personal/Build Orchestrator/schedules/2026-W99-legacy-mixed.yaml' would contain "
        "builds resolving to ['edge-trading-system', 'other-project'] — one schedule file must be "
        "single-project; use separate schedule files with cross-schedule dependencies"
    )

    def _validate_graph(nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None):
        # Identical message regardless of mode -- the mix is unchanged by
        # this edit (build_id's own project never changed), only its
        # severity bucket differs (warning at baseline, would-be-error when
        # proposed), which is exactly what Phase 1's compatibility rule must
        # detect and drop.
        return {
            "ok": mode == "compat_existing",
            "errors": [] if mode == "compat_existing" else [
                {"code": "mixed_project_schedule", "message": same_message,
                 "path": "Personal/Build Orchestrator/schedules/2026-W99-legacy-mixed.yaml"},
            ],
            "warnings": [
                {"code": "mixed_project_schedule", "message": same_message,
                 "path": "Personal/Build Orchestrator/schedules/2026-W99-legacy-mixed.yaml"},
            ] if mode == "compat_existing" else [],
        }

    monkeypatch.setattr(bo_contract, "validate_graph", _validate_graph)

    is_new, _ = vault.write_file_atomic(
        "Personal/Build Orchestrator/specs/rv6-target.md", "new body", tool="vault_write",
    )
    assert not is_new
    assert spec_file.read_text() == "new body"


def test_changed_mixed_project_set_still_blocks(monkeypatch, vault_dir):
    """Control for the above: if the proposed graph's mixed_project_schedule
    message DIFFERS from the baseline's (e.g. build_id's own project edit
    changed the resulting project set), the edit genuinely changed the
    condition and must still reject."""
    schedule_content = (
        "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# w\n\nbuilds:\n\n"
        "  - id: other-project-sibling\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/other-project-sibling.md\n"
        "  - id: rv6-target\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/rv6-target.md\n"
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-legacy-mixed2.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(schedule_content)

    spec_file = vault_dir / "Personal" / "Build Orchestrator" / "specs" / "rv6-target.md"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    spec_file.write_text("old body")

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"rv6-target": "pending"}})
    monkeypatch.setattr(
        bo_contract, "preflight_spec_validate",
        lambda build_id, spec_markdown, spec_path, mode="compat_existing", timeout=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"rv6-target": "Personal/Build Orchestrator/schedules/2026-W99-legacy-mixed2.yaml"}},
    )

    def _validate_graph(nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None):
        if mode == "compat_existing":
            return {
                "ok": False,
                "errors": [],
                "warnings": [{"code": "mixed_project_schedule", "message": "resolving to ['edge-trading-system', 'other-project']",
                              "path": "Personal/Build Orchestrator/schedules/2026-W99-legacy-mixed2.yaml"}],
            }
        return {
            "ok": False,
            "errors": [{"code": "mixed_project_schedule", "message": "resolving to ['other-project', 'third-project']",
                        "path": "Personal/Build Orchestrator/schedules/2026-W99-legacy-mixed2.yaml"}],
            "warnings": [],
        }

    monkeypatch.setattr(bo_contract, "validate_graph", _validate_graph)

    with pytest.raises(bo_guard.BOGuardError):
        vault.write_file_atomic("Personal/Build Orchestrator/specs/rv6-target.md", "new body", tool="vault_write")
    assert spec_file.read_text() == "old body"


def test_dependency_cycle_always_blocks_regardless_of_baseline_match(monkeypatch, vault_dir):
    """Control: Phase 1's baseline-vs-proposed compatibility carve-out is
    scoped ONLY to mixed_project_schedule -- a dependency_cycle finding must
    remain unconditionally blocking even when byte-identical between
    baseline and proposed."""
    schedule_content = (
        "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# w\n\nbuilds:\n\n"
        "  - id: rv6-target\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/rv6-target.md\n"
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-cycle.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(schedule_content)

    spec_file = vault_dir / "Personal" / "Build Orchestrator" / "specs" / "rv6-target.md"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    spec_file.write_text("old body")

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"rv6-target": "pending"}})
    monkeypatch.setattr(
        bo_contract, "preflight_spec_validate",
        lambda build_id, spec_markdown, spec_path, mode="compat_existing", timeout=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"rv6-target": "Personal/Build Orchestrator/schedules/2026-W99-cycle.yaml"}},
    )

    same_cycle_message = "dependency cycle: rv6-target -> rv6-target"

    def _validate_graph(nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None):
        return {
            "ok": mode == "compat_existing",
            "errors": [] if mode == "compat_existing" else [{"code": "dependency_cycle", "message": same_cycle_message}],
            "warnings": [{"code": "dependency_cycle", "message": same_cycle_message}] if mode == "compat_existing" else [],
        }

    monkeypatch.setattr(bo_contract, "validate_graph", _validate_graph)

    with pytest.raises(bo_guard.BOGuardError):
        vault.write_file_atomic("Personal/Build Orchestrator/specs/rv6-target.md", "new body", tool="vault_write")
    assert spec_file.read_text() == "old body"


# --------------------------------------------------------------------------
# vault-bo-authoring-cache-remediation-v7: sibling content-shape bleed
# (residual finding flagged, not fixed, by vault-bo-authoring-guard-
# remediation-v6's real-corpus report -- HI5-3's filter only scoped the
# three DB-state codes; any OTHER per-node error code, e.g.
# missing_summary_instruction from preflight_spec_validate, still bled
# through unfiltered when attributed to a sibling).
# --------------------------------------------------------------------------


def _mixed_schedule_and_target(vault_dir, schedule_name):
    schedule_content = (
        "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# w\n\nbuilds:\n\n"
        "  - id: dirty-sibling\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/dirty-sibling.md\n"
        "  - id: rv6-target\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/rv6-target.md\n"
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / schedule_name
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(schedule_content)

    spec_file = vault_dir / "Personal" / "Build Orchestrator" / "specs" / "rv6-target.md"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    spec_file.write_text("old body")
    return spec_file


def test_sibling_own_content_shape_defect_does_not_block_benign_edit(monkeypatch, vault_dir):
    """A clean target must not be blocked merely because a sibling elsewhere
    in the same schedule has its own unrelated content-shape defect (e.g.
    missing_summary_instruction) -- that is the sibling's own problem, not
    the target's, matching the mixed_project/terminal_id_reuse precedent."""
    spec_file = _mixed_schedule_and_target(vault_dir, "2026-W99-dirty-sibling.yaml")

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"rv6-target": "pending"}})
    monkeypatch.setattr(
        bo_contract, "preflight_spec_validate",
        lambda build_id, spec_markdown, spec_path, mode="compat_existing", timeout=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"rv6-target": "Personal/Build Orchestrator/schedules/2026-W99-dirty-sibling.yaml"}},
    )
    monkeypatch.setattr(
        bo_contract, "validate_graph",
        lambda nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None: {
            "ok": False,
            "errors": [{
                "code": "missing_summary_instruction",
                "message": "spec body has no /tmp/cc-summary-{build_id}.txt instruction",
                "build_id": "dirty-sibling",
            }],
            "warnings": [],
        },
    )

    is_new, _ = vault.write_file_atomic(
        "Personal/Build Orchestrator/specs/rv6-target.md", "new body", tool="vault_write",
    )
    assert not is_new
    assert spec_file.read_text() == "new body"


def test_own_content_shape_defect_from_graph_check_still_blocks(monkeypatch, vault_dir):
    """Control: the same code, attributed to the build actually being
    rewritten rather than a sibling, must still reject -- the v7 fix drops
    sibling-attributed noise, not build_id's own findings."""
    spec_file = _mixed_schedule_and_target(vault_dir, "2026-W99-dirty-self.yaml")

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {"rv6-target": "pending"}})
    monkeypatch.setattr(
        bo_contract, "preflight_spec_validate",
        lambda build_id, spec_markdown, spec_path, mode="compat_existing", timeout=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(
        bo_contract, "preflight_source_schedule",
        lambda build_ids, timeout=None: {"results": {"rv6-target": "Personal/Build Orchestrator/schedules/2026-W99-dirty-self.yaml"}},
    )
    monkeypatch.setattr(
        bo_contract, "validate_graph",
        lambda nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None: {
            "ok": False,
            "errors": [{
                "code": "missing_summary_instruction",
                "message": "spec body has no /tmp/cc-summary-{build_id}.txt instruction",
                "build_id": "rv6-target",
            }],
            "warnings": [],
        },
    )

    with pytest.raises(bo_guard.BOGuardError):
        vault.write_file_atomic("Personal/Build Orchestrator/specs/rv6-target.md", "new body", tool="vault_write")
    assert spec_file.read_text() == "old body"


# --------------------------------------------------------------------------
# MD5-1: move/delete of an un-ingested-but-referenced spec.
# --------------------------------------------------------------------------


def test_move_of_uningested_but_referenced_spec_is_rejected(monkeypatch, vault_dir):
    schedule_content = (
        "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# w\n\nbuilds:\n\n"
        "  - id: rv6-uningested\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/rv6-uningested.md\n"
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-md1.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(schedule_content)

    spec_file = vault_dir / "Personal" / "Build Orchestrator" / "specs" / "rv6-uningested.md"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    spec_file.write_text("body")

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    # No DB row at all -- this is exactly the un-ingested case.
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {bid: None for bid in build_ids}})

    with pytest.raises(bo_guard.BOGuardError):
        vault.move_path("Personal/Build Orchestrator/specs/rv6-uningested.md", "Scratch/moved-out.md")
    assert spec_file.exists()


def test_delete_of_uningested_but_referenced_spec_is_rejected(monkeypatch, vault_dir):
    schedule_content = (
        "---\ntype: schedule\nproject: edge-trading-system\n---\n\n# w\n\nbuilds:\n\n"
        "  - id: rv6-uningested-del\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/rv6-uningested-del.md\n"
    )
    schedule_file = vault_dir / "Personal" / "Build Orchestrator" / "schedules" / "2026-W99-md1-del.yaml"
    schedule_file.parent.mkdir(parents=True, exist_ok=True)
    schedule_file.write_text(schedule_content)

    spec_file = vault_dir / "Personal" / "Build Orchestrator" / "specs" / "rv6-uningested-del.md"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    spec_file.write_text("body")

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {bid: None for bid in build_ids}})

    with pytest.raises(bo_guard.BOGuardError):
        vault.delete_path("Personal/Build Orchestrator/specs/rv6-uningested-del.md")
    assert spec_file.exists()


def test_move_of_uningested_and_unreferenced_spec_is_allowed(monkeypatch, vault_dir):
    """Sanity control: MD5-1's new check must not block an un-ingested spec
    that genuinely isn't referenced by any on-disk schedule."""
    spec_file = vault_dir / "Personal" / "Build Orchestrator" / "specs" / "rv6-orphan.md"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    spec_file.write_text("body")

    monkeypatch.setenv("BO_PATH_GUARD_MODE", "enforce")
    monkeypatch.setattr(bo_contract, "preflight_ids", lambda build_ids, timeout=None: {"results": {bid: None for bid in build_ids}})

    assert vault.move_path("Personal/Build Orchestrator/specs/rv6-orphan.md", "Scratch/moved-out.md") is True


# --------------------------------------------------------------------------
# LO5-1: vault-root containment on _read_vault_text.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("traversal", [
    "../../../../../../../../etc/hostname",
    "/etc/hostname",
])
def test_read_vault_text_refuses_out_of_vault_paths(vault_dir, traversal):
    assert bo_guard._read_vault_text(traversal) is None


def test_read_vault_text_still_reads_in_vault_paths(vault_dir):
    (vault_dir / "in-vault.md").write_text("hello")
    assert bo_guard._read_vault_text("in-vault.md") == "hello"


def test_crafted_spec_path_in_schedule_entry_cannot_read_outside_vault(vault_dir):
    """End-to-end: a schedule entry's spec_path is caller-influenced content
    -- it must not be used to read arbitrary host files into the node graph
    the adapter evaluates."""
    builds = [{"id": "x", "spec_path": "../../../../../../../../etc/hostname"}]
    nodes = bo_guard._nodes_for_schedule_builds(builds, "Personal/Build Orchestrator/schedules/x.yaml", None)
    assert nodes[0]["spec_markdown"] == ""
