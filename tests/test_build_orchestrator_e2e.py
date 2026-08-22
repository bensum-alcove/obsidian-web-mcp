"""Real end-to-end tests for the BO authoring tools against the actual
authoring_contract.py subprocess CLI (no mocking) -- skipped automatically if
that repo isn't present on this host. Complements test_build_orchestrator_tools.py
(which mocks bo_contract to test this repo's own orchestration logic in
isolation) by proving the real wiring actually works.
"""

import json
from pathlib import Path

import pytest

from obsidian_vault_mcp import config
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
