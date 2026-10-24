"""Test fixtures for the Obsidian vault MCP server."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def vault_dir(tmp_path, monkeypatch):
    """Create a temporary vault directory with sample files."""
    vault = tmp_path / "test-vault"
    vault.mkdir()

    # test-note.md with frontmatter
    (vault / "test-note.md").write_text(
        "---\nstatus: active\ntype: note\n---\n\nThis is a test note with some content.\n"
    )

    # subfolder/nested-note.md with frontmatter
    subfolder = vault / "subfolder"
    subfolder.mkdir()
    (subfolder / "nested-note.md").write_text(
        "---\nstatus: draft\ntype: client-hub\nclient: TestCorp\n---\n\nNested note content.\n"
    )

    # no-frontmatter.md
    (vault / "no-frontmatter.md").write_text("Just plain text, no frontmatter here.\n")

    # .obsidian/config.json (should be excluded)
    obsidian_dir = vault / ".obsidian"
    obsidian_dir.mkdir()
    (obsidian_dir / "config.json").write_text('{"theme": "dark"}')

    # Set environment variable for config module
    monkeypatch.setenv("VAULT_PATH", str(vault))
    monkeypatch.setenv("VAULT_MCP_TOKEN", "test-token-12345")

    # Reload config to pick up new env var
    import obsidian_vault_mcp.config as config
    config.VAULT_PATH = Path(str(vault))

    yield vault


@pytest.fixture(scope="session")
def _authoring_db_template(tmp_path_factory):
    """A new synthetic BO schema, never a copy of the live database."""
    import subprocess
    from obsidian_vault_mcp import bo_contract, config
    directory = tmp_path_factory.mktemp('authoring-db-template')
    script = '''
import sys
sys.path.insert(0, sys.argv[1])
import db
db._DEBUG_LOG = sys.argv[2]
db.init_db()
'''
    subprocess.run(
        [config.BO_AUTHORING_CONTRACT_PYTHON, '-c', script,
         str(Path(bo_contract.resolve_authoring_contract_path()).parent), str(directory/'debug.log')],
        env=dict(os.environ, PYTEST_CURRENT_TEST='authoring-fixture', BO_TEST_DB_DIR=str(directory)),
        check=True, capture_output=True, text=True,
    )
    return directory / 'orchestrator.db'


@pytest.fixture
def authoring_db(_authoring_db_template, tmp_path, monkeypatch):
    import shutil
    directory = tmp_path / 'bo-db'
    directory.mkdir()
    target = directory / 'orchestrator.db'
    shutil.copyfile(_authoring_db_template, target)
    monkeypatch.setenv('BO_TEST_DB_DIR', str(directory))
    return target
