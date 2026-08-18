import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "contradiction_lint.py"


def _load():
    spec = importlib.util.spec_from_file_location("contradiction_lint", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_report_parses_and_quotes_generated_date():
    module = _load()
    existing = (
        "---\r\ngenerated: '2026-01-01'\r\ntype: report\r\n---\r\n"
        "Generated: 2026-01-01\r\n\r\nManual body\r\n"
    )
    section = module.SYSTEM_FACTS_MARKER + "\n\n## Automated\n"

    result = module.build_report(
        existing, section, datetime(2026, 8, 15, tzinfo=timezone.utc)
    )

    assert "generated: '2026-08-15'\r\n" in result
    assert "Manual body" in result
    assert result.count(module.SYSTEM_FACTS_MARKER) == 1


def test_build_report_refuses_invalid_frontmatter():
    module = _load()
    with pytest.raises(ValueError, match="unparseable"):
        module.build_report(
            "---\ngenerated: [broken\n---\nbody\n",
            "section",
            datetime(2026, 8, 15, tzinfo=timezone.utc),
        )


def test_doc_updated_date_rejects_non_date(capsys):
    module = _load()
    assert module._doc_updated_date("---\nupdated: unknown\n---\nbody\n") is None
    assert "invalid SYSTEM-FACTS updated date" in capsys.readouterr().err


def _write_record(dir_path, name, frontmatter):
    path = dir_path / name
    path.write_text(f"---\n{frontmatter}\n---\n\nbody\n", encoding="utf-8")
    return path


def test_canonical_state_alignment_clean_reference_has_no_findings(tmp_path):
    module = _load()
    _write_record(
        tmp_path,
        "foo-bar.md",
        "type: canonical-state\ncomponent_id: foo-bar\nstate: active\n"
        "content_updated: '2026-08-01'\nverified_at: '2026-08-18'\nsource: 'x'\n",
    )
    findings = module.check_canonical_state_alignment(
        {"infrastructure.md": "see `foo-bar` canonical record"},
        tmp_path,
        datetime(2026, 8, 18, tzinfo=timezone.utc),
    )
    assert findings == []


def test_canonical_state_alignment_ignores_unreferenced_records(tmp_path):
    module = _load()
    _write_record(
        tmp_path,
        "unmentioned.md",
        "type: canonical-state\ncomponent_id: unmentioned\nstate: active\n"
        "content_updated: '2026-08-01'\nverified_at: '2026-08-18'\nsource: 'x'\n",
    )
    findings = module.check_canonical_state_alignment(
        {"infrastructure.md": "no pointers here at all"},
        tmp_path,
        datetime(2026, 8, 18, tzinfo=timezone.utc),
    )
    assert findings == []


def test_canonical_state_alignment_flags_referenced_but_superseded(tmp_path):
    module = _load()
    _write_record(
        tmp_path,
        "old-thing.md",
        "type: canonical-state\ncomponent_id: old-thing\nstate: retired\n"
        "content_updated: '2026-08-01'\nverified_at: '2026-08-01'\nsource: 'x'\n"
        "superseded_by: new-thing\n",
    )
    findings = module.check_canonical_state_alignment(
        {"SYSTEM-FACTS.md": "still points at `old-thing` here"},
        tmp_path,
        datetime(2026, 8, 18, tzinfo=timezone.utc),
    )
    assert len(findings) == 1
    assert findings[0]["issue"] == "referenced_but_missing"
    assert findings[0]["component_id"] == "old-thing"
    assert findings[0]["source"] == "SYSTEM-FACTS.md"


def test_canonical_state_alignment_flags_duplicate_authority(tmp_path):
    module = _load()
    fm = (
        "type: canonical-state\ncomponent_id: dup-thing\nstate: active\n"
        "content_updated: '2026-08-01'\nverified_at: '2026-08-18'\nsource: 'x'\n"
    )
    _write_record(tmp_path, "dup-thing.md", fm)
    _write_record(tmp_path, "dup-thing--variant.md", fm)
    findings = module.check_canonical_state_alignment(
        {"infrastructure.md": "references `dup-thing` here"},
        tmp_path,
        datetime(2026, 8, 18, tzinfo=timezone.utc),
    )
    assert len(findings) == 1
    assert findings[0]["issue"] == "duplicate_authority"


def test_canonical_state_alignment_flags_stale_record(tmp_path):
    module = _load()
    _write_record(
        tmp_path,
        "aging-thing.md",
        "type: canonical-state\ncomponent_id: aging-thing\nstate: active\n"
        "content_updated: '2026-01-01'\nverified_at: '2026-01-01'\nsource: 'x'\n",
    )
    findings = module.check_canonical_state_alignment(
        {"infrastructure.md": "see `aging-thing` for details"},
        tmp_path,
        datetime(2026, 8, 18, tzinfo=timezone.utc),
    )
    assert len(findings) == 1
    assert findings[0]["issue"] == "stale"


def test_render_canonical_state_section_empty_findings_says_so():
    module = _load()
    rendered = module.render_canonical_state_section([], datetime(2026, 8, 18, tzinfo=timezone.utc))
    assert "No findings" in rendered


def test_render_canonical_state_section_renders_table_for_findings():
    module = _load()
    findings = [
        {
            "component_id": "dup-thing",
            "source": "infrastructure.md",
            "issue": "duplicate_authority",
            "detail": "two current records",
        }
    ]
    rendered = module.render_canonical_state_section(findings, datetime(2026, 8, 18, tzinfo=timezone.utc))
    assert "dup-thing" in rendered
    assert "duplicate_authority" in rendered
