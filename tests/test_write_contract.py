"""Fixtures and behavioural contract tests for the write-contract gate."""

import json
import time

import pytest

from obsidian_vault_mcp import write_contract as wc
from obsidian_vault_mcp.tools.manage import vault_delete, vault_move
from obsidian_vault_mcp.tools.write import vault_batch_write, vault_patch_section, vault_write
from obsidian_vault_mcp.vault import move_path, resolve_vault_path, write_file_atomic


# --------------------------------------------------------------------------
# Mode plumbing
# --------------------------------------------------------------------------


def test_mode_defaults_to_shadow(monkeypatch):
    monkeypatch.delenv("VAULT_WRITE_CONTRACT_MODE", raising=False)
    assert wc.get_mode() == "shadow"


def test_unrecognised_mode_falls_back_to_shadow(monkeypatch):
    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE", "bogus")
    assert wc.get_mode() == "shadow"


def test_off_mode_runs_no_validators(monkeypatch):
    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE", "off")
    result = wc.evaluate_content(
        wc.WriteContext(path="x.md", old_content=None, new_content="---\nfoo: 1\nfoo: 2\n---\n")
    )
    assert result.issues == []


def test_shadow_mode_reports_but_never_blocks(monkeypatch):
    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE", "shadow")
    result = wc.evaluate_content(
        wc.WriteContext(path="x.md", old_content=None, new_content="---\nfoo: 1\nfoo: 2\n---\n")
    )
    assert any(i.rule_id == "frontmatter-parseable" for i in result.issues)
    assert result.blocked is False


def test_enforce_mode_only_blocks_rules_marked_enforced(monkeypatch):
    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE", "enforce")
    # frontmatter-parseable is shipped shadow-only (not yet proven against a
    # real-vault scan) -- a reject-severity issue from it must not block here.
    assert not wc.is_rule_enforced("frontmatter-parseable")
    result = wc.evaluate_content(
        wc.WriteContext(path="x.md", old_content=None, new_content="---\nfoo: 1\nfoo: 2\n---\n")
    )
    assert any(i.rule_id == "frontmatter-parseable" for i in result.issues)
    assert result.blocked is False


def test_enforce_mode_blocks_a_rule_explicitly_marked_enforced(monkeypatch):
    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE", "enforce")
    monkeypatch.setattr(wc, "_ENFORCED_RULES", {"frontmatter-parseable"})
    result = wc.evaluate_content(
        wc.WriteContext(path="x.md", old_content=None, new_content="---\nfoo: 1\nfoo: 2\n---\n")
    )
    assert result.blocked is True
    assert result.blocking_issues[0].rule_id == "frontmatter-parseable"


def test_validator_exception_never_breaks_or_blocks_a_write(monkeypatch):
    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE", "enforce")

    def boom(ctx):
        raise RuntimeError("validator bug")

    monkeypatch.setitem(wc._CONTENT_REGISTRY, "broken-rule", boom)
    monkeypatch.setattr(wc, "_ENFORCED_RULES", {"broken-rule"})
    result = wc.evaluate_content(wc.WriteContext(path="x.md", old_content=None, new_content="hello"))
    assert result.blocked is False


# --------------------------------------------------------------------------
# frontmatter-parseable
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "---\nfoo: 1\nfoo: 2\n---\nbody\n",  # duplicate keys
        "---\nfoo: 1\nbody without closing delimiter\n",  # unterminated
        "---\n- a\n- b\n---\nbody\n",  # non-mapping top level
        "﻿---\nfoo: 1\n---\nbody\n",  # BOM before frontmatter
    ],
    ids=["duplicate-keys", "unterminated", "non-mapping", "bom"],
)
def test_frontmatter_parseable_rejects_adversarial_yaml(content):
    issues = wc._validate_frontmatter_parseable(
        wc.WriteContext(path="x.md", old_content=None, new_content=content)
    )
    assert len(issues) == 1
    assert issues[0].rule_id == "frontmatter-parseable"
    assert issues[0].severity == "reject"


@pytest.mark.parametrize(
    "content",
    [
        "---\n# a comment explaining this field\nfoo: 1\n---\nbody\n",
        "---\nbase: &b\n  x: 1\nref:\n  <<: *b\n  y: 2\n---\nbody\n",  # anchors/merge keys
        "---\ntype: some-completely-unknown-type\nfoo: bar\n---\nbody\n",  # unknown type
        "Just a legacy note with no frontmatter block at all.\n",  # valid legacy note
        "---\ntitle: 'quoted scalar'\ntags: [a, b, c]\n---\n\n# Heading\n\nBody.\n",
    ],
    ids=["comment", "anchor-merge-key", "unknown-type", "no-frontmatter", "normal-note"],
)
def test_frontmatter_parseable_accepts_legitimate_content(content):
    issues = wc._validate_frontmatter_parseable(
        wc.WriteContext(path="x.md", old_content=None, new_content=content)
    )
    assert issues == []


# --------------------------------------------------------------------------
# frontmatter-dates-quoted
# --------------------------------------------------------------------------


def test_bare_date_is_flagged_advisory():
    issues = wc._validate_dates_quoted(
        wc.WriteContext(path="x.md", old_content=None, new_content="---\nupdated: 2026-08-17\n---\nbody\n")
    )
    assert len(issues) == 1
    assert issues[0].rule_id == "frontmatter-dates-quoted"
    assert issues[0].severity == "advisory"


def test_quoted_date_is_not_flagged():
    issues = wc._validate_dates_quoted(
        wc.WriteContext(path="x.md", old_content=None, new_content="---\nupdated: '2026-08-17'\n---\nbody\n")
    )
    assert issues == []


def test_nested_bare_date_is_flagged():
    content = "---\nmeta:\n  history:\n    - date: 2026-01-01\n---\nbody\n"
    issues = wc._validate_dates_quoted(wc.WriteContext(path="x.md", old_content=None, new_content=content))
    assert any(i.rule_id == "frontmatter-dates-quoted" for i in issues)


# --------------------------------------------------------------------------
# protected-read-policy-full-rewrite
# --------------------------------------------------------------------------

_READ_POLICY_OLD = "---\nread_policy: section-only\n---\n\n## A\n\nfirst\n\n## B\n\nsecond\n"


def test_full_rewrite_of_read_policy_file_is_rejected():
    issues = wc._validate_read_policy_full_rewrite(
        wc.WriteContext(
            path="big.md",
            old_content=_READ_POLICY_OLD,
            new_content="---\nread_policy: section-only\n---\n\ncompletely different\n",
            tool="vault_write",
        )
    )
    assert len(issues) == 1
    assert issues[0].rule_id == "protected-read-policy-full-rewrite"
    assert issues[0].severity == "reject"


def test_section_patch_of_read_policy_file_is_allowed():
    # vault_patch_section can legitimately rewrite most of a section's text;
    # it's the tool identity that exempts it, not how similar the text stays.
    new_content = _READ_POLICY_OLD.replace("first", "completely reworded paragraph")
    issues = wc._validate_read_policy_full_rewrite(
        wc.WriteContext(path="big.md", old_content=_READ_POLICY_OLD, new_content=new_content, tool="vault_patch_section")
    )
    assert issues == []


def test_read_policy_rule_ignores_non_whole_file_tools_even_with_totally_different_body():
    for tool in ["vault_append", "vault_str_replace", "vault_batch_str_replace", "vault_batch_frontmatter_update", ""]:
        issues = wc._validate_read_policy_full_rewrite(
            wc.WriteContext(path="big.md", old_content=_READ_POLICY_OLD, new_content="anything at all", tool=tool)
        )
        assert issues == [], tool


def test_read_policy_rule_ignores_files_without_the_marker():
    old = "---\nstatus: active\n---\n\nold body\n"
    issues = wc._validate_read_policy_full_rewrite(
        wc.WriteContext(path="normal.md", old_content=old, new_content="---\nstatus: active\n---\n\nbrand new body\n", tool="vault_write")
    )
    assert issues == []


def test_read_policy_rule_ignores_new_files():
    issues = wc._validate_read_policy_full_rewrite(
        wc.WriteContext(path="new.md", old_content=None, new_content="---\nread_policy: section-only\n---\nbody\n", tool="vault_write")
    )
    assert issues == []


def test_read_policy_rule_ignores_batch_write_when_content_unchanged():
    issues = wc._validate_read_policy_full_rewrite(
        wc.WriteContext(path="big.md", old_content=_READ_POLICY_OLD, new_content=_READ_POLICY_OLD, tool="vault_batch_write")
    )
    assert issues == []


# --------------------------------------------------------------------------
# unsafe-path-chars
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["a:b.md", "weird<name>.md", "notes/CON.md", "notes/con.txt", "pipe|file.md"],
)
def test_unsafe_path_chars_rejected(path):
    issues = wc._validate_unsafe_path_chars(wc.WriteContext(path=path, old_content=None, new_content="x"))
    assert issues, path
    assert all(i.rule_id == "unsafe-path-chars" for i in issues)


@pytest.mark.parametrize("path", ["notes/normal-file.md", "Clients/Acme Corp.md", "sub/dir/note.md"])
def test_normal_paths_accepted(path):
    issues = wc._validate_unsafe_path_chars(wc.WriteContext(path=path, old_content=None, new_content="x"))
    assert issues == []


# --------------------------------------------------------------------------
# unsafe-file-extension
# --------------------------------------------------------------------------


def test_unknown_extension_is_advisory_only():
    issues = wc._validate_file_extension(wc.WriteContext(path="script.exe", old_content=None, new_content="x"))
    assert len(issues) == 1
    assert issues[0].severity == "advisory"


def test_known_extensions_pass():
    for path in ["note.md", "board.canvas", "table.base", "data.json", "sheet.csv", "plain.txt"]:
        assert wc._validate_file_extension(wc.WriteContext(path=path, old_content=None, new_content="x")) == []


# --------------------------------------------------------------------------
# frontmatter-required-fields (data-driven, opt-in per vault)
# --------------------------------------------------------------------------


def test_required_fields_rule_is_silent_without_schema_rules_file(vault_dir):
    content = "---\ntype: build-spec\n---\nbody\n"
    issues = wc._validate_required_fields(wc.WriteContext(path="x.md", old_content=None, new_content=content))
    assert issues == []


def test_required_fields_rule_flags_missing_field_when_opted_in(vault_dir):
    (vault_dir / "_schema_rules.json").write_text(json.dumps({"build-spec": ["build_id", "tier"]}))
    content = "---\ntype: build-spec\nbuild_id: foo\n---\nbody\n"
    issues = wc._validate_required_fields(wc.WriteContext(path="x.md", old_content=None, new_content=content))
    assert len(issues) == 1
    assert issues[0].severity == "advisory"
    assert "tier" in issues[0].message


def test_required_fields_rule_passes_when_all_present(vault_dir):
    (vault_dir / "_schema_rules.json").write_text(json.dumps({"build-spec": ["build_id"]}))
    content = "---\ntype: build-spec\nbuild_id: foo\n---\nbody\n"
    issues = wc._validate_required_fields(wc.WriteContext(path="x.md", old_content=None, new_content=content))
    assert issues == []


def test_malformed_schema_rules_file_never_raises(vault_dir):
    (vault_dir / "_schema_rules.json").write_text("not json {{{")
    content = "---\ntype: build-spec\n---\nbody\n"
    issues = wc._validate_required_fields(wc.WriteContext(path="x.md", old_content=None, new_content=content))
    assert issues == []


# --------------------------------------------------------------------------
# protected-structural-file (move / delete)
# --------------------------------------------------------------------------


def test_protected_root_file_blocks_delete():
    issues = wc._validate_protected_root_file(wc.PathMutationContext(path="_SCHEMA.md", operation="delete"))
    assert len(issues) == 1
    assert issues[0].severity == "reject"


def test_protected_root_file_blocks_move():
    issues = wc._validate_protected_root_file(
        wc.PathMutationContext(path="_SCHEMA.md", operation="move", destination="_SCHEMA_old.md")
    )
    assert len(issues) == 1


def test_unprotected_file_move_and_delete_pass():
    assert wc._validate_protected_root_file(wc.PathMutationContext(path="notes/x.md", operation="delete")) == []
    assert (
        wc._validate_protected_root_file(
            wc.PathMutationContext(path="notes/x.md", operation="move", destination="notes/y.md")
        )
        == []
    )


# --------------------------------------------------------------------------
# End-to-end: rejected writes leave source bytes (and mtime) untouched
# --------------------------------------------------------------------------


def test_rejected_write_leaves_source_bytes_and_mtime_unchanged(vault_dir, monkeypatch):
    target = vault_dir / "protected.md"
    original_bytes = b"---\nfoo: 1\n---\n\noriginal body\n"
    target.write_bytes(original_bytes)
    original_mtime = target.stat().st_mtime_ns

    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE", "enforce")
    monkeypatch.setattr(wc, "_ENFORCED_RULES", {"frontmatter-parseable"})

    time.sleep(0.01)
    result = json.loads(vault_write("protected.md", "---\nfoo: 1\nfoo: 2\n---\n\nnew body\n"))

    assert "error" in result
    assert target.read_bytes() == original_bytes
    assert target.stat().st_mtime_ns == original_mtime
    assert list(vault_dir.glob("*.tmp")) == []


def test_write_contract_error_is_a_value_error(vault_dir, monkeypatch):
    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE", "enforce")
    monkeypatch.setattr(wc, "_ENFORCED_RULES", {"frontmatter-parseable"})
    with pytest.raises(ValueError):
        write_file_atomic("x.md", "---\nfoo: 1\nfoo: 2\n---\n")


def test_enforced_rejection_surfaces_through_batch_write(vault_dir, monkeypatch):
    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE", "enforce")
    monkeypatch.setattr(wc, "_ENFORCED_RULES", {"frontmatter-parseable"})
    result = json.loads(vault_batch_write([{"path": "bad.md", "content": "---\nfoo: 1\nfoo: 2\n---\n"}]))
    assert result["failed"] == 1
    assert result["written"] == 0
    assert not (vault_dir / "bad.md").exists()


def test_enforced_protected_root_delete_blocked(vault_dir, monkeypatch):
    (vault_dir / "_SCHEMA.md").write_text("schema content\n")
    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE", "enforce")
    monkeypatch.setattr(wc, "_ENFORCED_RULES", {"protected-structural-file"})

    result = json.loads(vault_delete("_SCHEMA.md", confirm=True))
    assert "error" in result
    assert (vault_dir / "_SCHEMA.md").exists()


def test_enforced_protected_root_move_blocked(vault_dir, monkeypatch):
    (vault_dir / "_SCHEMA.md").write_text("schema content\n")
    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE", "enforce")
    monkeypatch.setattr(wc, "_ENFORCED_RULES", {"protected-structural-file"})

    result = json.loads(vault_move("_SCHEMA.md", "_SCHEMA_archived.md"))
    assert "error" in result
    assert (vault_dir / "_SCHEMA.md").exists()
    assert not (vault_dir / "_SCHEMA_archived.md").exists()


def test_shadow_mode_never_blocks_full_pipeline(vault_dir, monkeypatch):
    """Every rule fires in shadow mode (nothing enforced), but nothing is ever blocked."""
    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE", "shadow")
    result = json.loads(vault_write("shadow.md", "---\nfoo: 1\nfoo: 2\n---\n\nbody\n"))
    assert "error" not in result
    assert (vault_dir / "shadow.md").read_text().startswith("---\nfoo: 1\nfoo: 2\n---")
