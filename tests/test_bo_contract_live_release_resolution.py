"""Tests for bo_contract's live-BO-release path resolution
(vault-mcp-bo-adapter-live-contract-2026-09-28).

BO_AUTHORING_CONTRACT_PATH previously defaulted to a hard-coded/vendored
checkout path that could lag the actual deployed orchestrator release after a
BO deploy (reported schema v21 and rejected v22's model_probe_authority while
live BO was already on v22). These tests are hermetic: they point the
resolver at a temporary fake supervisord.conf tree via monkeypatch, never at
the real host config.
"""

import textwrap

import pytest

from obsidian_vault_mcp import bo_contract, config


def _write_supervisord_tree(tmp_path, conf_d_files):
    """Write a fake supervisord.conf + conf.d/*.conf tree. `conf_d_files` maps
    filename -> content. Returns the supervisord.conf path."""
    conf_d = tmp_path / "conf.d"
    conf_d.mkdir(exist_ok=True)
    for name, content in conf_d_files.items():
        (conf_d / name).write_text(textwrap.dedent(content), encoding="utf-8")
    supervisord_conf = tmp_path / "supervisord.conf"
    supervisord_conf.write_text(
        f"[include]\nfiles = {conf_d}/*.conf\n", encoding="utf-8",
    )
    return str(supervisord_conf)


@pytest.fixture(autouse=True)
def _clear_authoring_contract_override(monkeypatch):
    # None of these tests should ever fall through to whatever the real host
    # has configured via BO_AUTHORING_CONTRACT_PATH.
    monkeypatch.setattr(config, "BO_AUTHORING_CONTRACT_PATH", "")


def test_resolves_directory_from_orchestrator_program_section(tmp_path, monkeypatch):
    live_dir = tmp_path / "deployments" / "orchestrator-abc1234"
    live_dir.mkdir(parents=True)
    (live_dir / "authoring_contract.py").write_text("# fake adapter\n", encoding="utf-8")

    supervisord_conf = _write_supervisord_tree(tmp_path, {
        "orchestrator.conf": f"""
            [program:orchestrator]
            command=python3 {live_dir}/orchestrator.py
            directory={live_dir}
            """,
    })
    monkeypatch.setattr(bo_contract, "_supervisord_config_candidates", lambda: [supervisord_conf])

    resolved = bo_contract.resolve_authoring_contract_path()
    assert resolved == str(live_dir / "authoring_contract.py")


def test_explicit_override_wins_over_dynamic_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BO_AUTHORING_CONTRACT_PATH", "/explicit/authoring_contract.py")
    monkeypatch.setattr(
        bo_contract, "_supervisord_config_candidates",
        lambda: (_ for _ in ()).throw(AssertionError("should not consult supervisord when overridden")),
    )
    assert bo_contract.resolve_authoring_contract_path() == "/explicit/authoring_contract.py"


def test_fails_closed_when_no_supervisord_conf_found(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bo_contract, "_supervisord_config_candidates",
        lambda: [str(tmp_path / "does-not-exist.conf")],
    )
    with pytest.raises(bo_contract.BOContractError) as exc:
        bo_contract.resolve_authoring_contract_path()
    assert exc.value.code == "adapter_unresolved"


def test_fails_closed_when_orchestrator_section_missing(tmp_path, monkeypatch):
    supervisord_conf = _write_supervisord_tree(tmp_path, {
        "other.conf": """
            [program:something-else]
            command=/bin/true
            directory=/tmp
            """,
    })
    monkeypatch.setattr(bo_contract, "_supervisord_config_candidates", lambda: [supervisord_conf])
    with pytest.raises(bo_contract.BOContractError) as exc:
        bo_contract.resolve_authoring_contract_path()
    assert exc.value.code == "adapter_unresolved"


def test_fails_closed_when_orchestrator_section_is_ambiguous(tmp_path, monkeypatch):
    """Two conf.d files both defining [program:orchestrator] must never let
    the resolver silently pick one -- that would be as unsafe as guessing."""
    supervisord_conf = _write_supervisord_tree(tmp_path, {
        "a.conf": """
            [program:orchestrator]
            command=python3 /a/orchestrator.py
            directory=/a
            """,
        "b.conf": """
            [program:orchestrator]
            command=python3 /b/orchestrator.py
            directory=/b
            """,
    })
    monkeypatch.setattr(bo_contract, "_supervisord_config_candidates", lambda: [supervisord_conf])
    with pytest.raises(bo_contract.BOContractError) as exc:
        bo_contract.resolve_authoring_contract_path()
    assert exc.value.code == "adapter_unresolved"


def test_fails_closed_when_directory_option_missing(tmp_path, monkeypatch):
    supervisord_conf = _write_supervisord_tree(tmp_path, {
        "orchestrator.conf": """
            [program:orchestrator]
            command=python3 orchestrator.py
            """,
    })
    monkeypatch.setattr(bo_contract, "_supervisord_config_candidates", lambda: [supervisord_conf])
    with pytest.raises(bo_contract.BOContractError) as exc:
        bo_contract.resolve_authoring_contract_path()
    assert exc.value.code == "adapter_unresolved"


def test_authoring_field_projection_cache_invalidates_when_live_release_changes(tmp_path, monkeypatch):
    """The whole point of this fix: a BO deploy that moves the live release
    to a new directory must be picked up on the very next call, without
    requiring a vault MCP restart."""
    monkeypatch.setattr(bo_contract, "_authoring_fields_cache", None)
    monkeypatch.setattr(bo_contract, "_authoring_fields_cache_path", None)

    def make_release(tag, schedule_key):
        release_dir = tmp_path / f"orchestrator-{tag}"
        release_dir.mkdir()
        (release_dir / "authoring_contract.py").write_text(
            f'_SPEC_FRONTMATTER_KEY_ORDER = ("build_id", "tier")\n'
            f'_SCHEDULE_ENTRY_KEY_ORDER = ("id", "{schedule_key}")\n'
            f"def render_spec(build_id, tier):\n    pass\n",
            encoding="utf-8",
        )
        return release_dir

    release_v1 = make_release("v1", "depends_on")
    supervisord_conf = _write_supervisord_tree(tmp_path, {
        "orchestrator.conf": f"""
            [program:orchestrator]
            command=python3 {release_v1}/orchestrator.py
            directory={release_v1}
            """,
    })
    monkeypatch.setattr(bo_contract, "_supervisord_config_candidates", lambda: [supervisord_conf])

    projection_v1 = bo_contract.authoring_field_projection()
    assert "depends_on" in projection_v1["schedule_entry_keys"]
    assert "model_probe_authority" not in projection_v1["schedule_entry_keys"]

    # Simulate a BO deploy: a new release directory, same conf.d file rewritten
    # to point at it (exactly what a real deploy does to orchestrator.conf).
    release_v2 = make_release("v2", "model_probe_authority")
    supervisord_conf_v2 = _write_supervisord_tree(tmp_path, {
        "orchestrator.conf": f"""
            [program:orchestrator]
            command=python3 {release_v2}/orchestrator.py
            directory={release_v2}
            """,
    })
    monkeypatch.setattr(bo_contract, "_supervisord_config_candidates", lambda: [supervisord_conf_v2])

    projection_v2 = bo_contract.authoring_field_projection()
    assert "model_probe_authority" in projection_v2["schedule_entry_keys"]
