"""Regression coverage for file modes across every public write tool."""

import json
import os
import stat

import pytest

from obsidian_vault_mcp.tools.write import (
    vault_append,
    vault_batch_frontmatter_update,
    vault_batch_str_replace,
    vault_batch_write,
    vault_patch_section,
    vault_str_replace,
    vault_write,
)
from obsidian_vault_mcp.vault import NEW_FILE_MODE, write_file_atomic


EXISTING = "---\nstatus: active\n---\n\n# Target\n\nold body\n"


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.parametrize(
    ("name", "invoke"),
    [
        ("vault_write", lambda: vault_write("mode.md", EXISTING.replace("old body", "new body"))),
        ("vault_append", lambda: vault_append("mode.md", "appended\n", ensure_newline=False)),
        ("vault_str_replace", lambda: vault_str_replace("mode.md", "old body", "new body")),
        ("vault_patch_section", lambda: vault_patch_section("mode.md", "# Target", "new body")),
        ("vault_batch_write", lambda: vault_batch_write([{"path": "mode.md", "content": "batch"}])),
        (
            "vault_batch_str_replace",
            lambda: vault_batch_str_replace(
                [{"path": "mode.md", "old_str": "old body", "new_str": "new body"}]
            ),
        ),
        (
            "vault_batch_frontmatter_update",
            lambda: vault_batch_frontmatter_update(
                [{"path": "mode.md", "fields": {"status": "done"}}]
            ),
        ),
    ],
)
def test_every_write_tool_preserves_existing_mode(vault_dir, name, invoke):
    target = vault_dir / "mode.md"
    target.write_text(EXISTING, encoding="utf-8")
    os.chmod(target, 0o6755)

    result = json.loads(invoke())

    assert "error" not in result, (name, result)
    assert _mode(target) == 0o6755, name


def test_new_atomic_file_is_signer_readable(vault_dir):
    write_file_atomic("new-mode.md", "content")

    mode = _mode(vault_dir / "new-mode.md")
    assert mode == NEW_FILE_MODE
    assert mode & stat.S_IROTH


def test_deleted_destination_falls_back_to_new_file_mode(vault_dir, monkeypatch):
    target = vault_dir / "race.md"
    target.write_text("old", encoding="utf-8")
    os.chmod(target, 0o6755)
    real_stat = type(target).stat
    calls = 0

    def deleting_stat(self, *args, **kwargs):
        nonlocal calls
        if self == target:
            calls += 1
            if calls == 2:
                target.unlink()
                raise FileNotFoundError(target)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(type(target), "stat", deleting_stat)
    is_new, _ = write_file_atomic("race.md", "new")

    assert is_new is True
    assert _mode(target) == NEW_FILE_MODE


def test_failed_replace_cleans_up_temp_file(vault_dir, monkeypatch):
    import obsidian_vault_mcp.vault as vault_module

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(vault_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_file_atomic("failure.md", "content")

    assert list(vault_dir.glob("*.tmp")) == []
