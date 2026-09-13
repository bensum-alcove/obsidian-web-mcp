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

from obsidian_vault_mcp import config
from obsidian_vault_mcp.bo_guard import schedule_builds_from_content
from obsidian_vault_mcp.tools import build_orchestrator as bo

_ADAPTER_PRESENT = Path(config.BO_AUTHORING_CONTRACT_PATH).exists()
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
