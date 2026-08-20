"""Tests for evals/run_eval_v3.py (build: vault-retrieval-eval-v3).

evals/ is not an installed package (pythonpath is src/ only, per
pyproject.toml), so it's imported here the same way run_eval_v3.py itself
loads run_eval.py -- by file path via importlib.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = REPO_ROOT / "evals"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EVALS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rev3(monkeypatch, tmp_path):
    """A fresh run_eval_v3 module instance per test, with HISTORY_V3_DIR /
    CORPUS_HASH_PATH / DEPLOY_GATES_PATH redirected into tmp_path so tests
    never touch this repo's real evals/history-v3/."""
    module = _load_module("run_eval_v3_test_instance", "run_eval_v3.py")
    history_dir = tmp_path / "history-v3"
    monkeypatch.setattr(module, "HISTORY_V3_DIR", history_dir)
    monkeypatch.setattr(module, "CORPUS_HASH_PATH", history_dir / "CORPUS_HASH")
    monkeypatch.setattr(module, "DEPLOY_GATES_PATH", history_dir / "deploy-gates.yaml")
    return module


# ---------------------------------------------------------------------------
# score_entry
# ---------------------------------------------------------------------------

def _entry(**overrides):
    base = {
        "question": "q",
        "category": "known_facts",
        "expected_paths": ["a.md"],
        "decoy_paths": [],
        "negative": False,
        "require_all": False,
    }
    base.update(overrides)
    return base


def test_score_entry_standard_hit_and_rr(rev3):
    entry = _entry(expected_paths=["a.md", "b.md"])
    result = rev3.score_entry(["x.md", "a.md", "y.md"], entry)
    assert result["hit"] == 1.0
    assert result["rr"] == pytest.approx(1 / 2)


def test_score_entry_miss(rev3):
    entry = _entry(expected_paths=["a.md"])
    result = rev3.score_entry(["x.md", "y.md", "z.md", "w.md", "v.md", "a.md"], entry)
    # a.md is at rank 6, outside R_AT_K=5 -> hit is 0, but rr still finds it beyond top-5
    assert result["hit"] == 0.0
    assert result["rr"] == pytest.approx(1 / 6)


def test_score_entry_source_correct_when_expected_ranks_first(rev3):
    entry = _entry(expected_paths=["a.md"], decoy_paths=["stale.md"])
    result = rev3.score_entry(["a.md", "stale.md"], entry)
    assert result["source_correct"] == 1.0


def test_score_entry_source_incorrect_when_decoy_ranks_first(rev3):
    entry = _entry(expected_paths=["a.md"], decoy_paths=["stale.md"])
    result = rev3.score_entry(["stale.md", "a.md"], entry)
    assert result["source_correct"] == 0.0


def test_score_entry_source_correct_none_without_decoy(rev3):
    entry = _entry(expected_paths=["a.md"], decoy_paths=[])
    result = rev3.score_entry(["a.md"], entry)
    assert result["source_correct"] is None


def test_score_entry_source_correct_none_when_neither_returned(rev3):
    entry = _entry(expected_paths=["a.md"], decoy_paths=["stale.md"])
    result = rev3.score_entry(["unrelated.md"], entry)
    assert result["source_correct"] is None


def test_score_entry_negative_correct_abstention(rev3):
    entry = _entry(expected_paths=[], negative=True, decoy_paths=["wrong.md"])
    result = rev3.score_entry(["something-else.md"], entry)
    assert result["source_correct"] == 1.0
    assert result["hit"] is None
    assert result["rr"] is None


def test_score_entry_negative_false_positive(rev3):
    entry = _entry(expected_paths=[], negative=True, decoy_paths=["wrong.md"])
    result = rev3.score_entry(["wrong.md", "other.md"], entry)
    assert result["source_correct"] == 0.0


def test_score_entry_negative_without_decoy_is_unscored(rev3):
    entry = _entry(expected_paths=[], negative=True, decoy_paths=[])
    result = rev3.score_entry(["anything.md"], entry)
    assert result["source_correct"] is None


def test_score_entry_synthesis_hit_requires_all(rev3):
    entry = _entry(expected_paths=["a.md", "b.md"], require_all=True)
    full = rev3.score_entry(["a.md", "b.md", "c.md"], entry)
    partial = rev3.score_entry(["a.md", "c.md", "d.md"], entry)
    assert full["synthesis_hit"] == 1.0
    assert partial["synthesis_hit"] == 0.0


def test_score_entry_no_synthesis_hit_when_not_require_all(rev3):
    entry = _entry(expected_paths=["a.md", "b.md"], require_all=False)
    result = rev3.score_entry(["a.md"], entry)
    assert result["synthesis_hit"] is None


# ---------------------------------------------------------------------------
# stratify_categories
# ---------------------------------------------------------------------------

def test_stratify_categories_flags_underpowered(rev3):
    eval_set = (
        [_entry(category="known_facts") for _ in range(5)]
        + [_entry(category="negative_unknown") for _ in range(2)]
    )
    strat = rev3.stratify_categories(eval_set)
    assert strat["known_facts"] == {"n": 5, "underpowered": False}
    assert strat["negative_unknown"] == {"n": 2, "underpowered": True}


# ---------------------------------------------------------------------------
# corpus hash freeze/drift
# ---------------------------------------------------------------------------

def test_compute_corpus_hash_deterministic(rev3):
    h1 = rev3.compute_corpus_hash("hello")
    h2 = rev3.compute_corpus_hash("hello")
    h3 = rev3.compute_corpus_hash("world")
    assert h1 == h2
    assert h1 != h3


def test_check_or_freeze_first_run_freezes(rev3):
    status = rev3.check_or_freeze_corpus_hash("abc123", freeze=False)
    assert status == {"frozen_hash": "abc123", "drift": False, "just_frozen": True}
    assert rev3.CORPUS_HASH_PATH.is_file()


def test_check_or_freeze_no_drift_on_matching_rerun(rev3):
    rev3.check_or_freeze_corpus_hash("abc123", freeze=False)
    status = rev3.check_or_freeze_corpus_hash("abc123", freeze=False)
    assert status == {"frozen_hash": "abc123", "drift": False, "just_frozen": False}


def test_check_or_freeze_detects_drift(rev3):
    rev3.check_or_freeze_corpus_hash("abc123", freeze=False)
    status = rev3.check_or_freeze_corpus_hash("different-hash", freeze=False)
    assert status["drift"] is True
    assert status["just_frozen"] is False
    assert status["frozen_hash"] == "abc123"


def test_check_or_freeze_explicit_freeze_overwrites(rev3):
    rev3.check_or_freeze_corpus_hash("abc123", freeze=False)
    status = rev3.check_or_freeze_corpus_hash("new-hash", freeze=True)
    assert status == {"frozen_hash": "new-hash", "drift": False, "just_frozen": True}


# ---------------------------------------------------------------------------
# contamination guard
# ---------------------------------------------------------------------------

def test_validate_no_excluded_paths_exits_on_expected_path_collision(rev3, tmp_path):
    eval_set = [_entry(expected_paths=[f"{rev3.REPORT_DIR_IN_VAULT}/2026-08-20-report.md"])]
    with pytest.raises(SystemExit):
        rev3._validate_no_excluded_paths_v3(eval_set, tmp_path)


def test_validate_no_excluded_paths_exits_on_decoy_path_collision(rev3, tmp_path):
    eval_set = [_entry(decoy_paths=[f"{rev3.REPORT_DIR_IN_VAULT}/2026-08-20-report.md"])]
    with pytest.raises(SystemExit):
        rev3._validate_no_excluded_paths_v3(eval_set, tmp_path)


def test_validate_no_excluded_paths_passes_for_clean_corpus(rev3, tmp_path):
    eval_set = [_entry(expected_paths=["BS 2nd Brain/Alcove/Infrastructure/SYSTEM-FACTS.md"])]
    # should not raise / exit
    rev3._validate_no_excluded_paths_v3(eval_set, tmp_path)


# ---------------------------------------------------------------------------
# rubric flagging
# ---------------------------------------------------------------------------

def test_needs_rubric_true_for_undecoyed_freshness_category(rev3):
    entry = _entry(category="recent_update_freshness", decoy_paths=[])
    assert rev3._needs_rubric(entry) is True


def test_needs_rubric_false_when_decoy_present(rev3):
    entry = _entry(category="recent_update_freshness", decoy_paths=["stale.md"])
    assert rev3._needs_rubric(entry) is False


def test_needs_rubric_false_for_non_rubric_category(rev3):
    entry = _entry(category="known_facts", decoy_paths=[])
    assert rev3._needs_rubric(entry) is False


def test_needs_rubric_false_for_negative(rev3):
    entry = _entry(category="archive_isolation", negative=True, expected_paths=[], decoy_paths=[])
    assert rev3._needs_rubric(entry) is False


def test_rubric_text_includes_category_note(rev3):
    entry = _entry(category_note="Check the frontmatter date.")
    assert "Check the frontmatter date." in rev3._rubric_text(entry)


# ---------------------------------------------------------------------------
# real corpus file integrity
# ---------------------------------------------------------------------------

def test_real_corpus_loads_and_validates(rev3):
    eval_set, raw = rev3.load_eval_set_v3()
    assert len(eval_set) >= 10
    assert isinstance(raw, str) and raw

    categories = {}
    for entry in eval_set:
        categories.setdefault(entry["category"], 0)
        categories[entry["category"]] += 1
    # every category the spec asks for must be represented
    required_categories = {
        "known_facts",
        "current_vs_stale",
        "canonical_contradiction_precedence",
        "entities_aliases",
        "exact_identifiers",
        "paraphrase",
        "cross_document",
        "negative_unknown",
        "recent_update_freshness",
        "archive_isolation",
    }
    assert required_categories.issubset(categories.keys())


def test_real_corpus_no_expected_or_decoy_path_collides_with_exclusions(rev3, tmp_path):
    eval_set, _raw = rev3.load_eval_set_v3()
    # tmp_path stands in for a vault root that doesn't contain any of these
    # files -- the frontmatter-exclude check short-circuits to False when the
    # file doesn't exist, so this only exercises the static glob-list check.
    rev3._validate_no_excluded_paths_v3(eval_set, tmp_path)


def test_real_corpus_negative_entries_have_empty_expected_paths(rev3):
    eval_set, _raw = rev3.load_eval_set_v3()
    for entry in eval_set:
        if entry["negative"]:
            assert entry["expected_paths"] == []
        else:
            assert entry["expected_paths"], f"non-negative entry with no expected_paths: {entry['question']!r}"
