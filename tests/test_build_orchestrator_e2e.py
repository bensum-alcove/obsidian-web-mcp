"""Real end-to-end tests for the BO authoring tools against the actual
authoring_contract.py subprocess CLI (no mocking) -- skipped automatically if
that repo isn't present on this host. Complements test_build_orchestrator_tools.py
(which mocks bo_contract to test this repo's own orchestration logic in
isolation) by proving the real wiring actually works.
"""

import json
from pathlib import Path

import frontmatter
import pytest

from obsidian_vault_mcp import bo_contract
from obsidian_vault_mcp.bo_guard import schedule_builds_from_content
from obsidian_vault_mcp.tools import build_orchestrator as bo


def _adapter_present() -> bool:
    try:
        return Path(bo_contract.resolve_authoring_contract_path()).exists()
    except bo_contract.BOContractError:
        return False


_ADAPTER_PRESENT = _adapter_present()
pytestmark = pytest.mark.skipif(
    not _ADAPTER_PRESENT, reason="build-orchestrator authoring_contract.py not present on this host"
)

SCHEDULE_PATH = "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml"
SCHEDULE_SEED = (
    "---\ntags:\n  - orchestrator\n  - schedule\ntype: schedule\nweek: '2026-W99'\n"
    "project: edge-trading-system\ncreated: '2026-08-17'\n---\n\n# 2026-W99 — scratch\n\nbuilds:\n"
)
EMPTY_FLOW_SCHEDULE_SEED = (
    "---\ntags:\n  - orchestrator\n  - schedule\ntype: schedule\nweek: '2026-W99'\n"
    "project: edge-trading-system\ncreated: '2026-08-17'\n---\n\n# 2026-W99 — scratch\n\nbuilds: []\n"
)


@pytest.fixture
def seeded_schedule(vault_dir):
    sched_dir = vault_dir / "Personal" / "Build Orchestrator" / "schedules"
    sched_dir.mkdir(parents=True)
    (sched_dir / "2026-W99-scratch.yaml").write_text(SCHEDULE_SEED)
    return vault_dir


def _build(build_id, **overrides):
    b = {
        "build_id": build_id, "title": "t", "body_markdown": "do the thing",
        "tier": "simple", "project": "edge-trading-system",
        "risk_domain": "observability", "blast_radius": "single-component",
        "reversible": True, "shadowable": True,
        # schema v6+ requires newly-authored specs to state completion intent
        # explicitly (missing_completion_intent) -- see
        # vault-checkout-remote-reconciliation-v1.
        "deployment_intent": "required",
    }
    b.update(overrides)
    return b


def test_real_validate_and_create_single_build(seeded_schedule):
    build = _build("scratch-real-e2e-single")
    validated = json.loads(bo.bo_validate_build_graph([build], SCHEDULE_PATH))
    assert validated["ok"] is True, validated
    assert not (seeded_schedule / "Personal/Build Orchestrator/specs/scratch-real-e2e-single.md").exists()

    created = json.loads(bo.bo_create_build(build, SCHEDULE_PATH))
    assert created["ok"] is True, created
    assert (seeded_schedule / "Personal/Build Orchestrator/specs/scratch-real-e2e-single.md").exists()
    schedule_content = (seeded_schedule / "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml").read_text()
    assert "id: scratch-real-e2e-single" in schedule_content


def test_real_create_preserves_authored_checkout_policy(seeded_schedule):
    build_id = "scratch-real-e2e-policy-authored"
    policy = "## Checkout write policy\n\n- Source edits allowed and required.  \n- Keep  spacing.\n\n"
    body = "Do the thing.\n\n" + policy + "## Next heading\nContinue."
    build = _build(
        build_id, body_markdown=body, deployment_intent="not_applicable",
        work_role="executor",
        completion_contract={
            "assertions": [
                {"type": "git_pushed", "push_required": True,
                 "head_matches": f"origin/candidate/{build_id}"},
                {"type": "summary_valid"},
                {"type": "checkout_no_new_dirt"},
            ],
            "waivers": [],
        },
    )
    result = json.loads(bo.bo_create_build(build, SCHEDULE_PATH))
    assert result["ok"] is True, result
    stored = (seeded_schedule / f"Personal/Build Orchestrator/specs/{build_id}.md").read_text()
    assert stored.count("## Checkout write policy") == 1
    assert stored.split("## Checkout write policy", 1)[1].split("## Next heading", 1)[0] == (
        policy.split("## Checkout write policy", 1)[1]
    )


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("implementation", "Source and test edits within this build's scope are allowed and required"),
        ("reviewer", "Do not create, modify, delete, rename, or move any file"),
        ("deploy", "Do not edit source or test files"),
    ],
)
def test_real_create_uses_build_type_checkout_default(seeded_schedule, kind, expected):
    build_id = f"scratch-real-e2e-policy-{kind}"
    fields = {
        "implementation": {
            "deployment_intent": "not_applicable", "work_role": "executor",
            "completion_contract": {
                "assertions": [
                    {"type": "git_pushed", "push_required": True,
                     "head_matches": f"origin/candidate/{build_id}"},
                    {"type": "summary_valid"},
                    {"type": "checkout_no_new_dirt"},
                ],
                "waivers": [],
            },
        },
        "reviewer": {"deployment_intent": "not_applicable", "work_role": "reviewer"},
        "deploy": {"deployment_intent": "required", "work_role": "executor"},
    }[kind]
    result = json.loads(bo.bo_create_build(_build(build_id, **fields), SCHEDULE_PATH))
    assert result["ok"] is True, result
    stored = (seeded_schedule / f"Personal/Build Orchestrator/specs/{build_id}.md").read_text()
    assert stored.count("## Checkout write policy") == 1
    assert expected in stored


def test_real_rejects_unknown_project_with_no_writes(seeded_schedule):
    build = _build("scratch-real-e2e-badproj", project="totally-unconfigured-project-xyz")
    result = json.loads(bo.bo_create_build(build, SCHEDULE_PATH))
    assert result["ok"] is False
    assert any(e["code"] == "unknown_project" for e in result["errors"])
    assert not (seeded_schedule / "Personal/Build Orchestrator/specs/scratch-real-e2e-badproj.md").exists()


def test_real_rejects_dependency_cycle(seeded_schedule):
    b1 = _build("scratch-real-e2e-cyc1", depends_on=["scratch-real-e2e-cyc2"])
    b2 = _build("scratch-real-e2e-cyc2", depends_on=["scratch-real-e2e-cyc1"])
    result = json.loads(bo.bo_create_chain([b1, b2], SCHEDULE_PATH))
    assert result["ok"] is False
    assert any(e["code"] == "dependency_cycle" for e in result["errors"])
    assert not (seeded_schedule / "Personal/Build Orchestrator/specs/scratch-real-e2e-cyc1.md").exists()


def test_real_forward_reference_chain_succeeds(seeded_schedule):
    b1 = _build("scratch-real-e2e-fwd1", depends_on=["scratch-real-e2e-fwd2"])
    b2 = _build("scratch-real-e2e-fwd2")
    result = json.loads(bo.bo_create_chain([b1, b2], SCHEDULE_PATH))
    assert result["ok"] is True, result
    assert (seeded_schedule / "Personal/Build Orchestrator/specs/scratch-real-e2e-fwd1.md").exists()
    assert (seeded_schedule / "Personal/Build Orchestrator/specs/scratch-real-e2e-fwd2.md").exists()


def test_real_review_gate_resources_and_deployment_intent_survive_create(seeded_schedule):
    """schema-parity fields (vault-checkout-remote-reconciliation-v1): a
    structured create must actually persist review_gate/resources/
    deployment_intent into the written spec (and resources into the written
    schedule entry too) -- not just accept them without error."""
    review_gate = {
        "artifact_path": "BS 2nd Brain/Alcove/Infrastructure/Hardening/Reviews/opus-review-scratch-real-e2e-schema.md",
        "required_verdict": "APPROVED",
        "required_status": "pass",
        "max_blocker_count": 0,
        "max_high_count": 0,
        "accepted_models": ["claude-opus-4-8"],
        "require_model_verified": True,
        "reviewed_sha_must_match": "current_head",
    }
    resources = [{"id": "repo:scratch-real-e2e-schema", "mode": "exclusive"}]
    build = _build(
        "scratch-real-e2e-schema",
        deployment_intent="required",
        review_gate=review_gate,
        resources=resources,
    )

    validated = json.loads(bo.bo_validate_build_graph([build], SCHEDULE_PATH))
    assert validated["ok"] is True, validated

    created = json.loads(bo.bo_create_build(build, SCHEDULE_PATH))
    assert created["ok"] is True, created

    spec_content = (
        seeded_schedule / "Personal/Build Orchestrator/specs/scratch-real-e2e-schema.md"
    ).read_text()
    parsed_spec = frontmatter.loads(spec_content)
    assert parsed_spec.metadata["deployment_intent"] == "required"
    assert parsed_spec.metadata["review_gate"]["required_verdict"] == "APPROVED"
    assert parsed_spec.metadata["resources"] == resources

    schedule_content = (
        seeded_schedule / "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml"
    ).read_text()
    entries = schedule_builds_from_content(schedule_content, source_name=SCHEDULE_PATH)
    entry = next(e for e in entries if e["id"] == "scratch-real-e2e-schema")
    assert entry["resources"] == resources


def test_real_bad_review_gate_shape_rejected_with_no_writes(seeded_schedule):
    """The adapter, not this repo, still owns review_gate's shape rules --
    a malformed block must fail validate_graph, proving this repo isn't
    silently accepting a value it never checks."""
    build = _build("scratch-real-e2e-badgate", review_gate={"artifact_path": 123})
    result = json.loads(bo.bo_create_build(build, SCHEDULE_PATH))
    assert result["ok"] is False
    assert any(e["code"] == "bad_review_gate_field" for e in result["errors"])
    assert not (seeded_schedule / "Personal/Build Orchestrator/specs/scratch-real-e2e-badgate.md").exists()


def test_real_empty_flow_list_schedule_creates_exactly_one_entry(seeded_schedule):
    sched_path = seeded_schedule / "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml"
    sched_path.write_text(EMPTY_FLOW_SCHEDULE_SEED)
    build = _build("scratch-real-e2e-empty-flow")
    created = json.loads(bo.bo_create_build(build, SCHEDULE_PATH))
    assert created["ok"] is True, created
    content = sched_path.read_text()
    entries = schedule_builds_from_content(content, source_name=SCHEDULE_PATH)
    matching = [e for e in entries if e["id"] == "scratch-real-e2e-empty-flow"]
    assert len(matching) == 1
    assert "builds: []" not in content


def test_real_empty_flow_list_chain_writes_entries_in_order(seeded_schedule):
    sched_path = seeded_schedule / "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml"
    sched_path.write_text(EMPTY_FLOW_SCHEDULE_SEED)
    b1 = _build("scratch-real-e2e-empty-a")
    b2 = _build("scratch-real-e2e-empty-b", depends_on=["scratch-real-e2e-empty-a"])
    result = json.loads(bo.bo_create_chain([b1, b2], SCHEDULE_PATH))
    assert result["ok"] is True, result
    entries = schedule_builds_from_content(sched_path.read_text(), source_name=SCHEDULE_PATH)
    assert [e["id"] for e in entries] == ["scratch-real-e2e-empty-a", "scratch-real-e2e-empty-b"]


def test_real_nonempty_schedule_still_appends(seeded_schedule):
    first = json.loads(bo.bo_create_build(_build("scratch-real-e2e-nonempty-1"), SCHEDULE_PATH))
    assert first["ok"] is True, first
    second = json.loads(bo.bo_create_build(_build("scratch-real-e2e-nonempty-2"), SCHEDULE_PATH))
    assert second["ok"] is True, second
    entries = schedule_builds_from_content(
        (seeded_schedule / "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml").read_text(),
        source_name=SCHEDULE_PATH,
    )
    assert [e["id"] for e in entries] == [
        "scratch-real-e2e-nonempty-1",
        "scratch-real-e2e-nonempty-2",
    ]


def test_real_activate_existing_spec_on_empty_flow_schedule(seeded_schedule):
    sched_path = seeded_schedule / "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml"
    sched_path.write_text(EMPTY_FLOW_SCHEDULE_SEED)
    build = _build("scratch-real-e2e-activate")
    created = json.loads(bo.bo_create_build(build, SCHEDULE_PATH))
    assert created["ok"] is True, created
    spec_path = seeded_schedule / "Personal/Build Orchestrator/specs/scratch-real-e2e-activate.md"
    spec_text = spec_path.read_text()
    sched_path.write_text(EMPTY_FLOW_SCHEDULE_SEED)
    result = json.loads(bo.bo_activate_existing_spec("scratch-real-e2e-activate", SCHEDULE_PATH))
    assert result["ok"] is True, result
    assert spec_path.read_text() == spec_text
    entries = schedule_builds_from_content(sched_path.read_text(), source_name=SCHEDULE_PATH)
    assert [e["id"] for e in entries] == ["scratch-real-e2e-activate"]


def test_real_mixed_project_on_existing_schedule_fails_closed(seeded_schedule):
    sched_path = seeded_schedule / "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml"
    sched_path.write_text(
        "---\ntags:\n  - orchestrator\n  - schedule\ntype: schedule\nweek: '2026-W99'\n"
        "project: edge-trading-system\ncreated: '2026-08-17'\n---\n\n# 2026-W99 — scratch\n\nbuilds:\n"
        "  - id: existing-other-project\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/existing-other-project.md\n"
        "    project: mcp-infrastructure\n"
    )
    original = sched_path.read_text()
    result = json.loads(bo.bo_create_build(_build("scratch-real-e2e-mixed"), SCHEDULE_PATH))
    assert result["ok"] is False
    assert any(e["code"] == "mixed_project_schedule" for e in result.get("errors", [])), result
    assert not (seeded_schedule / "Personal/Build Orchestrator/specs/scratch-real-e2e-mixed.md").exists()
    assert sched_path.read_text() == original


def test_real_goal_and_codex_routing_fields_round_trip(seeded_schedule):
    build_id = "scratch-real-e2e-v17-fields"
    build = _build(
        build_id,
        engine="codex",
        work_role="executor",
        codex_model="gpt-5.6-sol",
        codex_reasoning_effort="high",
        goal_id="scratch-real-e2e-v17-goal",
        goal_root_build_id=build_id,
        goal_completion=False,
    )
    validated = json.loads(bo.bo_validate_build_graph([build], SCHEDULE_PATH))
    assert validated["ok"] is True, validated
    created = json.loads(bo.bo_create_build(build, SCHEDULE_PATH))
    assert created["ok"] is True, created
    spec = frontmatter.loads(
        (seeded_schedule / f"Personal/Build Orchestrator/specs/{build_id}.md").read_text()
    )
    assert spec.metadata["goal_id"] == "scratch-real-e2e-v17-goal"
    assert spec.metadata["goal_root_build_id"] == build_id
    assert spec.metadata["goal_completion"] is False
    assert spec.metadata["work_role"] == "executor"
    assert spec.metadata["engine"] == "codex"
    assert spec.metadata["codex_model"] == "gpt-5.6-sol"
    assert spec.metadata["codex_reasoning_effort"] == "high"
    entries = schedule_builds_from_content(
        (seeded_schedule / "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml").read_text(),
        source_name=SCHEDULE_PATH,
    )
    entry = next(e for e in entries if e["id"] == build_id)
    assert entry["engine"] == "codex"
    assert entry["work_role"] == "executor"
    assert entry["codex_model"] == "gpt-5.6-sol"
    assert entry["codex_reasoning_effort"] == "high"


def test_real_invalid_engine_effort_combination_fails_closed(seeded_schedule):
    build = _build(
        "scratch-real-e2e-v17-badcombo",
        engine="cursor",
        codex_reasoning_effort="high",
    )
    result = json.loads(bo.bo_create_build(build, SCHEDULE_PATH))
    assert result["ok"] is False
    assert any(
        e["code"] == "codex_reasoning_effort_requires_codex_engine"
        for e in result.get("errors", [])
    ), result
    assert not (
        seeded_schedule / "Personal/Build Orchestrator/specs/scratch-real-e2e-v17-badcombo.md"
    ).exists()


def test_real_incomplete_goal_lineage_fails_closed(seeded_schedule):
    build = _build("scratch-real-e2e-v17-badgoal", goal_id="only-one-goal-field")
    result = json.loads(bo.bo_create_build(build, SCHEDULE_PATH))
    assert result["ok"] is False
    assert any(e["code"] == "incomplete_goal_authority" for e in result.get("errors", [])), result
    assert not (
        seeded_schedule / "Personal/Build Orchestrator/specs/scratch-real-e2e-v17-badgoal.md"
    ).exists()


def test_real_unknown_codex_model_fails_closed(seeded_schedule):
    build = _build(
        "scratch-real-e2e-v17-badmodel",
        engine="codex",
        codex_model="not-a-real-codex-model",
    )
    result = json.loads(bo.bo_create_build(build, SCHEDULE_PATH))
    assert result["ok"] is False
    assert any(e["code"] == "unknown_codex_model" for e in result.get("errors", [])), result
    assert not (
        seeded_schedule / "Personal/Build Orchestrator/specs/scratch-real-e2e-v17-badmodel.md"
    ).exists()
