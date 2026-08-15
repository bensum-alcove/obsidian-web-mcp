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
