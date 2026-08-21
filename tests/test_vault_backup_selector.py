"""Tests for scripts/vault_backup_selector.py (vault-chaos-recovery-suite build,
scenario #10: corrupt latest backup generation)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "vault_backup_selector.py"


@pytest.fixture(scope="module")
def selector_mod():
    spec = importlib.util.spec_from_file_location("vault_backup_selector", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(args, cwd):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "throwaway-vault-repo"
    root.mkdir()
    _git(["init", "-q"], root)
    _git(["config", "user.email", "chaos-drill@example.invalid"], root)
    _git(["config", "user.name", "chaos-drill"], root)
    return root


def _commit(repo_path, filename, content, message):
    (repo_path / filename).write_text(content)
    _git(["add", "-A"], repo_path)
    _git(["commit", "-q", "-m", message], repo_path)
    return _git(["rev-parse", "HEAD"], repo_path)


def test_selects_head_when_clean(selector_mod, repo):
    sha1 = _commit(repo, "note.md", "---\nstatus: active\n---\n\nGen 1.\n", "gen1")
    report = selector_mod.select_known_good_commit(repo)
    assert report["selected"] == sha1
    assert report["flagged_corrupt"] == []
    assert report["head_was_corrupt"] is False


def test_flags_corrupt_head_and_selects_known_good_predecessor(selector_mod, repo):
    sha_good = _commit(repo, "note.md", "---\nstatus: active\n---\n\nGen 1, good.\n", "gen1 good")
    _commit(repo, "note.md", "---\nstatus: active\n\nGen 2, unterminated frontmatter.\n", "gen2 corrupt")
    sha_head = _git(["rev-parse", "HEAD"], repo)

    report = selector_mod.select_known_good_commit(repo)

    assert report["head_was_corrupt"] is True
    assert report["flagged_corrupt"] == [sha_head]
    assert report["selected"] == sha_good
    assert report["generations_walked"] == 2


def test_never_returns_a_corrupt_generation_even_if_it_is_the_only_one(selector_mod, repo):
    _commit(repo, "note.md", "---\nstatus: [unterminated flow\n---\n\nonly gen, corrupt.\n", "only gen, corrupt")
    report = selector_mod.select_known_good_commit(repo)
    assert report["selected"] is None
    assert report["head_was_corrupt"] is True


def test_max_generations_bounds_how_far_back_it_walks(selector_mod, repo):
    _commit(repo, "note.md", "---\nstatus: active\n---\n\nGen 1, good.\n", "gen1 good")
    _commit(repo, "note.md", "---\nstatus: active\n\nGen 2, corrupt.\n", "gen2 corrupt")
    _commit(repo, "note.md", "---\nstatus: active\n\nGen 3, corrupt.\n", "gen3 corrupt")

    report = selector_mod.select_known_good_commit(repo, max_generations=2)
    assert report["generations_walked"] == 2
    assert report["selected"] is None  # the one good generation is out of the walked window


def test_check_generation_ignores_files_with_no_frontmatter_block(selector_mod, repo):
    sha = _commit(repo, "plain.md", "Just prose, no frontmatter at all.\n", "plain file")
    generation = selector_mod.check_generation(repo, sha)
    assert generation.clean
    assert generation.files_checked == 1


def test_never_mutates_the_repo_working_tree_or_history(selector_mod, repo):
    """Structural safety net: this module is read-only via `git show`/`git
    ls-tree`. Prove the working tree and HEAD are untouched by a run."""
    _commit(repo, "note.md", "---\nstatus: active\n---\n\nGen 1.\n", "gen1")
    before_head = _git(["rev-parse", "HEAD"], repo)
    before_status = _git(["status", "--porcelain"], repo)

    selector_mod.select_known_good_commit(repo)

    after_head = _git(["rev-parse", "HEAD"], repo)
    after_status = _git(["status", "--porcelain"], repo)
    assert before_head == after_head
    assert before_status == after_status == ""


def test_cli_main_writes_json_report(selector_mod, repo, tmp_path):
    _commit(repo, "note.md", "---\nstatus: active\n---\n\nGen 1.\n", "gen1")
    output = tmp_path / "report.json"
    old_argv = sys.argv
    try:
        sys.argv = ["vault_backup_selector.py", "--repo-path", str(repo), "--output", str(output)]
        exit_code = selector_mod.main()
    finally:
        sys.argv = old_argv
    assert exit_code == 0
    assert output.exists()
    import json
    report = json.loads(output.read_text())
    assert report["selected"] is not None
