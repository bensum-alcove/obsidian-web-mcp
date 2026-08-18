"""Tests for obsidian_vault_mcp.canonical_state (build: vault-canonical-state-model)."""

from pathlib import Path

import pytest

from obsidian_vault_mcp.canonical_state import (
    DuplicateAuthorityError,
    load_all_records,
    parse_record,
    resolve_current_state,
    scan_duplicate_authority,
    validate_frontmatter,
)


def _write(dir_path: Path, name: str, frontmatter: str) -> Path:
    path = dir_path / name
    path.write_text(f"---\n{frontmatter}\n---\n\nbody\n", encoding="utf-8")
    return path


VALID_FM = (
    "type: canonical-state\n"
    "component_id: vault-lock\n"
    "state: active\n"
    "content_updated: '2026-08-18'\n"
    "verified_at: '2026-08-18'\n"
    "source: 'commit c8989a0'\n"
)


def test_valid_record_parses_with_no_errors(tmp_path):
    path = _write(tmp_path, "vault-lock.md", VALID_FM)
    record, errors = parse_record(path)
    assert errors == []
    assert record is not None
    assert record.component_id == "vault-lock"
    assert record.state == "active"
    assert record.is_current


@pytest.mark.parametrize(
    "missing_field",
    ["type", "component_id", "state", "content_updated", "verified_at", "source"],
)
def test_missing_required_field_is_flagged(missing_field):
    metadata = {
        "type": "canonical-state",
        "component_id": "vault-lock",
        "state": "active",
        "content_updated": "2026-08-18",
        "verified_at": "2026-08-18",
        "source": "commit c8989a0",
    }
    del metadata[missing_field]
    errors = validate_frontmatter(metadata, expected_component_id="vault-lock")
    assert any(missing_field in e for e in errors)


# -- freshness semantics -----------------------------------------------------


def test_updated_alone_is_not_sufficient_freshness_signal():
    """`updated` (file-write metadata) is not a required or validated field --
    a record with a stale `updated` but current content_updated/verified_at is
    still valid, proving freshness is decided by the semantic fields, not by
    file-write metadata."""
    metadata = {
        "type": "canonical-state",
        "component_id": "vault-lock",
        "state": "active",
        "content_updated": "2026-08-18",
        "verified_at": "2026-08-18",
        "source": "commit c8989a0",
        "updated": "2020-01-01",  # ancient file-write date, irrelevant to state
    }
    assert validate_frontmatter(metadata, expected_component_id="vault-lock") == []


def test_verified_before_content_updated_is_rejected():
    metadata = {
        "type": "canonical-state",
        "component_id": "vault-lock",
        "state": "active",
        "content_updated": "2026-08-18",
        "verified_at": "2026-01-01",  # can't verify a fact before it became true
        "source": "commit c8989a0",
    }
    errors = validate_frontmatter(metadata, expected_component_id="vault-lock")
    assert any("predates" in e for e in errors)


def test_invalid_date_format_is_rejected():
    metadata = {
        "type": "canonical-state",
        "component_id": "vault-lock",
        "state": "active",
        "content_updated": "not-a-date",
        "verified_at": "2026-08-18",
        "source": "commit c8989a0",
    }
    errors = validate_frontmatter(metadata, expected_component_id="vault-lock")
    assert any("content_updated" in e for e in errors)


# -- supersession -------------------------------------------------------------


def test_superseded_record_excluded_from_resolution(tmp_path):
    _write(
        tmp_path,
        "vault-lock.md",
        VALID_FM + "superseded_by: vault-lock-v2\n",
    )
    _write(
        tmp_path,
        "vault-lock--v2.md",
        "type: canonical-state\n"
        "component_id: vault-lock\n"
        "state: hardened\n"
        "content_updated: '2026-08-18'\n"
        "verified_at: '2026-08-18'\n"
        "source: 'opus-review-phase2-write-integrity-v4'\n"
        "supersedes: vault-lock\n",
    )

    resolved = resolve_current_state("vault-lock", tmp_path)
    assert resolved is not None
    assert resolved.state == "hardened"
    assert resolved.component_id == "vault-lock"

    records, errors = load_all_records(tmp_path)
    assert errors == []
    assert len(records) == 2
    current = [r for r in records if r.is_current]
    assert len(current) == 1
    assert current[0].state == "hardened"


def test_no_current_record_resolves_to_none(tmp_path):
    _write(tmp_path, "orphan.md", VALID_FM + "superseded_by: nothing-real\n")
    assert resolve_current_state("vault-lock", tmp_path) is None


# -- invalid duplicate authority / two current records for one component -----


def test_two_current_records_for_one_component_raises(tmp_path):
    _write(tmp_path, "vault-lock--a.md", VALID_FM)
    _write(tmp_path, "vault-lock--b.md", VALID_FM.replace("state: active", "state: broken"))

    with pytest.raises(DuplicateAuthorityError) as excinfo:
        resolve_current_state("vault-lock", tmp_path)
    assert excinfo.value.component_id == "vault-lock"
    assert len(excinfo.value.records) == 2


def test_duplicate_authority_scanner_flags_component(tmp_path):
    _write(tmp_path, "vault-lock--a.md", VALID_FM)
    _write(tmp_path, "vault-lock--b.md", VALID_FM)
    _write(tmp_path, "hot-md-curation.md", VALID_FM.replace("vault-lock", "hot-md-curation"))

    duplicates = scan_duplicate_authority(tmp_path)
    assert set(duplicates.keys()) == {"vault-lock"}
    assert len(duplicates["vault-lock"]) == 2


def test_scanner_is_report_only_and_does_not_touch_disk(tmp_path):
    path_a = _write(tmp_path, "vault-lock--a.md", VALID_FM)
    path_b = _write(tmp_path, "vault-lock--b.md", VALID_FM)
    before_a, before_b = path_a.read_text(), path_b.read_text()

    scan_duplicate_authority(tmp_path)

    assert path_a.read_text() == before_a
    assert path_b.read_text() == before_b


# -- derived-view conflict ----------------------------------------------------


def test_component_id_mismatched_with_filename_is_derived_view_conflict(tmp_path):
    """The record's declared identity (frontmatter component_id) must agree
    with the identity a reader derives from its location (filename stem) --
    two independent "views" of the same record disagreeing is exactly the
    kind of silent-drift bug a single free-text prose file could never catch."""
    path = _write(tmp_path, "vault-lock.md", VALID_FM.replace(
        "component_id: vault-lock", "component_id: hot-md-curation"
    ))
    record, errors = parse_record(path)
    assert record is None
    assert any("derived-view conflict" in e for e in errors)


def test_malformed_record_does_not_block_scanning_the_rest(tmp_path):
    _write(tmp_path, "broken.md", VALID_FM.replace(
        "component_id: vault-lock", "component_id: hot-md-curation"
    ))
    _write(tmp_path, "vault-lock.md", VALID_FM)

    records, errors = load_all_records(tmp_path)
    assert len(records) == 1
    assert records[0].component_id == "vault-lock"
    assert len(errors) == 1
    assert "derived-view conflict" in errors[0]
