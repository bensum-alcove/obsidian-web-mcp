"""Tests for vault-retrieval-r5-085-v1's eval-report-directory exclusion.

Retrieval-eval report files (evals/run_eval_v3.py's dated reports) quote every
benchmark question verbatim as rubric/top_result text, so leaving them in
ordinary retrieval makes them the single strongest keyword match for a large
fraction of the very questions they document -- diagnosed as the direct cause
of over half this build's frozen-v3 misses before this fix. Any real user
asking one of these documented questions in production hits the same
contamination, so this is a genuine retrieval-quality fix, not a benchmark
edit.
"""

import json

from obsidian_vault_mcp import config
from obsidian_vault_mcp.tools.query import vault_query
from obsidian_vault_mcp.tools.search import vault_search, _excluded_dirs_for_scope


def test_eval_report_dir_names_in_retrieval_excluded_dirs():
    assert config.EVAL_REPORT_DIR_NAMES <= config.RETRIEVAL_EXCLUDED_DIRS
    assert "retrieval-eval" in config.RETRIEVAL_EXCLUDED_DIRS
    assert "retrieval-eval-v3" in config.RETRIEVAL_EXCLUDED_DIRS


def test_eval_report_dirs_not_in_base_excluded_dirs():
    """Same precedent as _scratch: hidden from ordinary retrieval only, still
    vault_list/frontmatter-index visible -- an operator can still browse past
    reports."""
    assert "retrieval-eval" not in config.EXCLUDED_DIRS
    assert "retrieval-eval-v3" not in config.EXCLUDED_DIRS


def test_vault_search_excludes_eval_report_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)

    report_dir = tmp_path / "Infrastructure" / "retrieval-eval-v3"
    report_dir.mkdir(parents=True)
    (report_dir / "2026-08-22-report.md").write_text(
        "uniquecontaminationmarker appears in this rubric quote.\n"
    )
    (tmp_path / "real-answer.md").write_text("uniquecontaminationmarker is the real content.\n")

    result = json.loads(vault_search("uniquecontaminationmarker"))
    paths = [m["path"] for m in result.get("matches", result.get("results", []))]
    assert not any("retrieval-eval-v3" in p for p in paths)
    assert any("real-answer.md" in p for p in paths)


def test_vault_query_excludes_eval_report_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)

    report_dir = tmp_path / "Infrastructure" / "retrieval-eval"
    report_dir.mkdir(parents=True)
    (report_dir / "2026-08-15-report.md").write_text(
        "distinctivequerysentinel shows up verbatim in this old report.\n"
    )
    (tmp_path / "real-answer.md").write_text("distinctivequerysentinel is the real content.\n")

    result = json.loads(vault_query("distinctivequerysentinel"))
    paths = [r["path"] for r in result["results"]]
    assert not any("retrieval-eval" in p for p in paths)
    assert any("real-answer.md" in p for p in paths)


def test_excluded_dirs_for_scope_still_permits_explicit_scoped_search(tmp_path, monkeypatch):
    """Same escape hatch as _scratch: an explicit, scoped request INTO the
    report directory must still see it, matching the existing rationale for
    _excluded_dirs_for_scope."""
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    scoped_path = tmp_path / "retrieval-eval-v3"
    scoped_path.mkdir()
    scoped = _excluded_dirs_for_scope(scoped_path)
    assert "retrieval-eval-v3" not in scoped
