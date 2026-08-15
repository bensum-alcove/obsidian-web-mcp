import importlib.util
from datetime import date
from pathlib import Path


AUDIT_PATH = Path("/home/ben_sum/vault-audit.py")


def _load_audit():
    spec = importlib.util.spec_from_file_location("vault_audit", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_autofix_uses_parser_quotes_date_and_preserves_body(tmp_path):
    audit = _load_audit()
    target = tmp_path / "note.md"
    body = "\n# Body\n\n---\n"
    target.write_text("---\nupdated: '2026-01-01'\ntags: [one]\n---" + body)
    item = {
        "rp": "note.md",
        "fp": str(target),
        "upd_date": date(2026, 1, 1),
        "mtime_date": date(2026, 8, 15),
    }

    fixed, skipped, errored = audit.autofix_stale_frontmatter([item])

    assert fixed and not skipped and not errored
    changed = target.read_text()
    assert "updated: '2026-08-15'\n" in changed
    assert "\n---\n# Body" in changed
    assert changed.endswith(body)


def test_autofix_skips_invalid_frontmatter_loudly(tmp_path):
    audit = _load_audit()
    target = tmp_path / "broken.md"
    original = "---\nupdated: [broken\n---\nbody\n"
    target.write_text(original)
    item = {
        "rp": "broken.md",
        "fp": str(target),
        "upd_date": date(2026, 1, 1),
        "mtime_date": date(2026, 8, 15),
    }

    fixed, skipped, errored = audit.autofix_stale_frontmatter([item])

    assert not fixed and not skipped
    assert "FRONTMATTER ERROR" in errored[0]
    assert target.read_text() == original
