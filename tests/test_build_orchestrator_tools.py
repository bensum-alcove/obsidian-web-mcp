"""Tests for tools/build_orchestrator.py -- bo_validate_build_graph, bo_create_build,
bo_create_chain.

bo_contract's check_version/render_graph/validate_graph are monkeypatched with
deterministic fakes throughout (this suite's job is to prove THIS module's
orchestration -- atomic write ordering, overwrite protection, fail-closed
adapter handling -- not to re-prove authoring_contract's own schema rules,
already covered by build-orchestrator's own 795-test suite). A couple of true
end-to-end tests against the real adapter close the loop, skipped if it's
absent from this host.
"""

import json
from pathlib import Path

import pytest

from obsidian_vault_mcp import bo_contract, config
from obsidian_vault_mcp.models import BOCreateBuildInput, BOCreateChainInput
from obsidian_vault_mcp.tools import build_orchestrator as bo

SCHEDULE_PATH = "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml"

SCHEDULE_SEED = (
    "---\ntags:\n  - orchestrator\n  - schedule\ntype: schedule\nweek: '2026-W99'\n"
    "project: edge-trading-system\ncreated: '2026-08-17'\n---\n\n# 2026-W99 — scratch\n\nbuilds:\n"
)


def _fake_render_graph(specs, timeout=None):
    rendered = {}
    for item in specs:
        bid = item["build_id"]
        entry = item["schedule_entry"]
        spec_lines = [
            "---", f"build_id: {bid}", f"tier: {item['tier']}", f"project: {item['project']}",
            f"status: {item.get('status', 'ready')}", "---", "", f"# {bid}", "", item["body_markdown"], "",
            f"Write a summary to /tmp/cc-summary-{bid}.txt where the FIRST LINE is exactly: {bid}.",
        ]
        spec_md = "\n".join(spec_lines) + "\n"
        entry_lines = [
            f"  - id: {bid}", f"    title: {entry['title']}",
            f"    description: {entry.get('description') or entry['title']}",
            f"    run_when: {entry.get('run_when') or 'no deps'}",
            f"    tier: {entry['tier']}",
            f"    depends_on: {entry.get('depends_on') or []}", f"    spec_path: {entry['spec_path']}",
        ]
        rendered[bid] = {"spec": spec_md, "schedule_entry": "\n".join(entry_lines) + "\n"}
    return {"rendered": rendered}


def _ok_validate_graph(nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None):
    return {"ok": True, "errors": [], "warnings": []}


@pytest.fixture(autouse=True)
def _patch_contract(monkeypatch):
    monkeypatch.setattr(bo_contract, "check_version", lambda timeout=None: {"schema_version": 6, "contract_version": "1.0.0"})
    monkeypatch.setattr(bo_contract, "render_graph", _fake_render_graph)


@pytest.fixture
def seeded_schedule(vault_dir):
    sched_dir = vault_dir / "Personal" / "Build Orchestrator" / "schedules"
    sched_dir.mkdir(parents=True)
    (sched_dir / "2026-W99-scratch.yaml").write_text(SCHEDULE_SEED)
    return vault_dir


def _build(build_id="scratch-1", **overrides):
    b = {
        "build_id": build_id, "title": "t", "body_markdown": "do the thing",
        "tier": "simple", "project": "edge-trading-system",
        "risk_domain": "observability", "blast_radius": "single-component",
        "reversible": True, "shadowable": True,
    }
    b.update(overrides)
    return b


def test_validate_reports_ok_true_and_canonical_graph(monkeypatch, seeded_schedule):
    monkeypatch.setattr(bo_contract, "validate_graph", _ok_validate_graph)
    result = json.loads(bo.bo_validate_build_graph([_build()], SCHEDULE_PATH))
    assert result["ok"] is True
    assert result["schema_version"] == 6
    assert result["canonical_graph"][0]["build_id"] == "scratch-1"
    assert result["canonical_graph"][0]["spec_path"] == "Personal/Build Orchestrator/specs/scratch-1.md"


def test_validate_never_writes_anything(monkeypatch, seeded_schedule):
    monkeypatch.setattr(bo_contract, "validate_graph", _ok_validate_graph)
    bo.bo_validate_build_graph([_build()], SCHEDULE_PATH)
    assert not (seeded_schedule / "Personal/Build Orchestrator/specs/scratch-1.md").exists()


def test_validate_reports_errors_when_adapter_rejects(monkeypatch, seeded_schedule):
    monkeypatch.setattr(
        bo_contract, "validate_graph",
        lambda nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None: {
            "ok": False, "errors": [{"code": "unknown_project", "message": "x"}], "warnings": [],
        },
    )
    result = json.loads(bo.bo_validate_build_graph([_build()], SCHEDULE_PATH))
    assert result["ok"] is False
    assert result["errors"][0]["code"] == "unknown_project"


def test_create_build_writes_spec_then_schedule(monkeypatch, seeded_schedule):
    monkeypatch.setattr(bo_contract, "validate_graph", _ok_validate_graph)
    result = json.loads(bo.bo_create_build(_build(), SCHEDULE_PATH))
    assert result["ok"] is True
    spec_file = seeded_schedule / "Personal/Build Orchestrator/specs/scratch-1.md"
    assert spec_file.exists()
    schedule_content = (seeded_schedule / "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml").read_text()
    assert "id: scratch-1" in schedule_content
    assert result["created"][0]["spec_path"] == "Personal/Build Orchestrator/specs/scratch-1.md"
    assert result["activation"]["schedule_path"] == SCHEDULE_PATH


def test_create_build_writes_nothing_when_validation_fails(monkeypatch, seeded_schedule):
    monkeypatch.setattr(
        bo_contract, "validate_graph",
        lambda nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None: {
            "ok": False, "errors": [{"code": "dependency_cycle", "message": "x -> y -> x"}], "warnings": [],
        },
    )
    result = json.loads(bo.bo_create_build(_build(), SCHEDULE_PATH))
    assert result["ok"] is False
    assert result["activated"] is False
    assert not (seeded_schedule / "Personal/Build Orchestrator/specs/scratch-1.md").exists()
    schedule_content = (seeded_schedule / "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml").read_text()
    assert schedule_content == SCHEDULE_SEED  # byte-identical -- nothing touched it


def test_create_build_fails_closed_when_adapter_unavailable(monkeypatch, seeded_schedule):
    def boom(timeout=None):
        raise bo_contract.BOContractError("adapter_missing", "not found")

    monkeypatch.setattr(bo_contract, "check_version", boom)
    result = json.loads(bo.bo_create_build(_build(), SCHEDULE_PATH))
    assert result["ok"] is False
    assert result["code"] == "adapter_missing"
    assert result["activated"] is False
    assert not (seeded_schedule / "Personal/Build Orchestrator/specs/scratch-1.md").exists()


def test_create_build_refuses_to_overwrite_existing_spec(monkeypatch, seeded_schedule):
    monkeypatch.setattr(bo_contract, "validate_graph", _ok_validate_graph)
    specs_dir = seeded_schedule / "Personal/Build Orchestrator/specs"
    specs_dir.mkdir(parents=True)
    (specs_dir / "scratch-1.md").write_text("---\nbuild_id: scratch-1\n---\nalready here\n")
    result = json.loads(bo.bo_create_build(_build(), SCHEDULE_PATH))
    assert result["ok"] is False
    assert "overwrite" in result["error"]
    assert (specs_dir / "scratch-1.md").read_text() == "---\nbuild_id: scratch-1\n---\nalready here\n"


def test_create_build_errors_cleanly_when_schedule_path_missing(monkeypatch, seeded_schedule):
    monkeypatch.setattr(bo_contract, "validate_graph", _ok_validate_graph)
    result = json.loads(bo.bo_create_build(_build(), "Personal/Build Orchestrator/schedules/does-not-exist.yaml"))
    assert result["ok"] is False
    assert "does not exist" in result["error"]
    assert not (seeded_schedule / "Personal/Build Orchestrator/specs/scratch-1.md").exists()


def test_create_build_never_creates_a_new_schedule_file(monkeypatch, vault_dir):
    """v1 scope: schedule_path must already exist -- matches build_generator.generate_build()."""
    monkeypatch.setattr(bo_contract, "validate_graph", _ok_validate_graph)
    result = json.loads(bo.bo_create_build(_build(), "Personal/Build Orchestrator/schedules/brand-new.yaml"))
    assert result["ok"] is False
    assert not (vault_dir / "Personal/Build Orchestrator/schedules/brand-new.yaml").exists()


def test_partial_failure_leaves_schedule_unactivated_and_specs_orphaned(monkeypatch, seeded_schedule):
    """Schedule write fails after spec write succeeds -- the spec is an orphan
    (inert, not referenced by any schedule entry), never a half-activated graph."""
    monkeypatch.setattr(bo_contract, "validate_graph", _ok_validate_graph)

    real_write = bo.write_file_atomic
    call_count = {"n": 0}

    def flaky_write(path, content, **kwargs):
        call_count["n"] += 1
        if "schedules" in path:
            raise ValueError("simulated schedule write failure")
        return real_write(path, content, **kwargs)

    monkeypatch.setattr(bo, "write_file_atomic", flaky_write)
    result = json.loads(bo.bo_create_build(_build(), SCHEDULE_PATH))
    assert result["ok"] is False
    assert result["orphaned_specs"] == ["Personal/Build Orchestrator/specs/scratch-1.md"]
    assert (seeded_schedule / "Personal/Build Orchestrator/specs/scratch-1.md").exists()  # orphaned but present
    schedule_content = (seeded_schedule / "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml").read_text()
    assert schedule_content == SCHEDULE_SEED  # schedule never activated


def test_create_chain_forward_reference_writes_all_specs_and_one_schedule_entry_each(monkeypatch, seeded_schedule):
    monkeypatch.setattr(bo_contract, "validate_graph", _ok_validate_graph)
    b1 = _build("chain-1", depends_on=["chain-2"])
    b2 = _build("chain-2")
    result = json.loads(bo.bo_create_chain([b1, b2], SCHEDULE_PATH))
    assert result["ok"] is True
    assert {c["build_id"] for c in result["created"]} == {"chain-1", "chain-2"}
    assert (seeded_schedule / "Personal/Build Orchestrator/specs/chain-1.md").exists()
    assert (seeded_schedule / "Personal/Build Orchestrator/specs/chain-2.md").exists()
    schedule_content = (seeded_schedule / "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml").read_text()
    assert "id: chain-1" in schedule_content
    assert "id: chain-2" in schedule_content


def test_create_chain_writes_nothing_if_any_build_fails_validation(monkeypatch, seeded_schedule):
    monkeypatch.setattr(
        bo_contract, "validate_graph",
        lambda nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None: {
            "ok": False, "errors": [{"code": "unknown_dependency", "message": "x", "build_id": "chain-1"}], "warnings": [],
        },
    )
    result = json.loads(bo.bo_create_chain([_build("chain-1", depends_on=["nonexistent"]), _build("chain-2")], SCHEDULE_PATH))
    assert result["ok"] is False
    assert not (seeded_schedule / "Personal/Build Orchestrator/specs/chain-1.md").exists()
    assert not (seeded_schedule / "Personal/Build Orchestrator/specs/chain-2.md").exists()


def test_mode_is_passed_through_to_validate_graph(monkeypatch, seeded_schedule):
    captured = {}

    def capture(nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None):
        captured["mode"] = mode
        return {"ok": True, "errors": [], "warnings": []}

    monkeypatch.setattr(bo_contract, "validate_graph", capture)
    bo.bo_validate_build_graph([_build()], SCHEDULE_PATH, mode="compat_existing")
    assert captured["mode"] == "compat_existing"


def test_new_ids_passed_to_validate_graph_is_the_proposed_build_ids(monkeypatch, seeded_schedule):
    captured = {}

    def capture(nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None):
        captured["new_ids"] = new_ids
        return {"ok": True, "errors": [], "warnings": []}

    monkeypatch.setattr(bo_contract, "validate_graph", capture)
    bo.bo_validate_build_graph([_build("scratch-1"), _build("scratch-2")], SCHEDULE_PATH)
    assert captured["new_ids"] == ["scratch-1", "scratch-2"]


def test_existing_schedule_entries_are_included_as_graph_context(monkeypatch, seeded_schedule):
    """B2: pre-existing entries already in schedule_path must be included in
    the graph passed to validate_graph, not just the newly-proposed nodes --
    otherwise cross-node checks (mixed project, duplicate id) never see them."""
    sched_path = seeded_schedule / "Personal/Build Orchestrator/schedules/2026-W99-scratch.yaml"
    sched_path.write_text(
        SCHEDULE_SEED.rstrip("\n") + "\n"
        "  - id: existing-build\n    title: t\n    description: t\n    run_when: x\n    tier: simple\n"
        "    depends_on: []\n    spec_path: Personal/Build Orchestrator/specs/existing-build.md\n"
        "    project: some-other-project\n"
    )
    captured = {}

    def capture(nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None):
        captured["node_ids"] = sorted(n["build_id"] for n in nodes)
        return {"ok": True, "errors": [], "warnings": []}

    monkeypatch.setattr(bo_contract, "validate_graph", capture)
    bo.bo_validate_build_graph([_build("scratch-1")], SCHEDULE_PATH)
    assert captured["node_ids"] == ["existing-build", "scratch-1"]


# --------------------------------------------------------------------------
# B3: compat_existing must not be selectable on a create/mutation path
# --------------------------------------------------------------------------


def test_bo_create_build_input_rejects_a_mode_field():
    """codex-review-bo-authoring-contract-v1, B3: 'Both BOCreateBuildInput and
    BOCreateChainInput accept caller-selected mode: compat_existing ... The
    create endpoints can therefore opt new artifacts out of strict-new
    validation.' mode is no longer a field on these input models at all --
    extra="forbid" means supplying it is a validation error, not silently
    ignored."""
    with pytest.raises(Exception):
        BOCreateBuildInput(build=_build(), schedule_path=SCHEDULE_PATH, mode="compat_existing")


def test_bo_create_chain_input_rejects_a_mode_field():
    with pytest.raises(Exception):
        BOCreateChainInput(builds=[_build()], schedule_path=SCHEDULE_PATH, mode="compat_existing")


def test_bo_create_build_python_function_no_longer_accepts_mode():
    """Belt-and-suspenders: the underlying tool function's signature itself
    has no mode parameter, so even a caller bypassing the Pydantic model
    can't pass one through."""
    import inspect
    assert "mode" not in inspect.signature(bo.bo_create_build).parameters
    assert "mode" not in inspect.signature(bo.bo_create_chain).parameters


def test_bo_create_build_always_validates_strict_new_even_for_legacy_shaped_build(monkeypatch, seeded_schedule):
    """A build missing v6 risk fields (the legacy/compat_existing-only shape)
    must be rejected by bo_create_build -- there is no longer any way to
    route a create call through compat_existing leniency."""
    captured = {}

    def capture(nodes, mode="strict_new", config_override=None, new_ids=None, timeout=None):
        captured["mode"] = mode
        return {"ok": False, "errors": [{"code": "missing_risk_fields", "message": "x"}], "warnings": []}

    monkeypatch.setattr(bo_contract, "validate_graph", capture)
    legacy_build = _build(risk_domain=None, blast_radius=None, reversible=None, shadowable=None)
    result = json.loads(bo.bo_create_build(legacy_build, SCHEDULE_PATH))
    assert captured["mode"] == "strict_new"
    assert result["ok"] is False

