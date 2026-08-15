from datetime import date, datetime, timezone

import pytest

from obsidian_vault_mcp.frontmatter_safe import (
    FrontmatterError,
    parse_frontmatter,
    update_frontmatter_field,
)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_changed_frontmatter_preserves_body_and_quotes_date(newline):
    body = f"Body{newline}---{newline}tail{newline}"
    content = f"---{newline}updated: 2026-01-01{newline}---{newline}{body}"
    changed = update_frontmatter_field(content, "updated", "2026-08-15")

    assert changed.endswith(body)
    assert f"updated: '2026-08-15'{newline}" in changed
    assert parse_frontmatter(changed).metadata["updated"] == "2026-08-15"


def test_date_objects_normalize_to_quoted_strings():
    content = "---\nupdated: old\ncreated: old\n---\nbody"
    changed = update_frontmatter_field(content, "updated", date(2026, 8, 15))
    changed = update_frontmatter_field(
        changed, "created", datetime(2026, 8, 15, 3, 4, tzinfo=timezone.utc)
    )

    assert "updated: '2026-08-15'" in changed
    assert "created: '2026-08-15T03:04:00+00:00'" in changed


def test_changed_field_preserves_comments_anchors_and_presentation():
    content = (
        "---\n"
        "# human rationale\n"
        "defaults: &defaults {one: 1, two: 2}\n"
        "copy: *defaults\n"
        "updated: 2026-01-01 # reviewed date\n"
        "---\nbody\n"
    )
    changed = update_frontmatter_field(content, "updated", "2026-08-15")

    assert "# human rationale" in changed
    assert "&defaults {one: 1, two: 2}" in changed
    assert "copy: *defaults" in changed
    assert "updated: '2026-08-15' # reviewed date" in changed


def test_duplicate_keys_are_rejected_not_collapsed():
    content = "---\naliases: [first]\naliases: [second]\nupdated: old\n---\nbody\n"
    with pytest.raises(FrontmatterError, match="duplicate key"):
        update_frontmatter_field(content, "updated", "2026-08-15")


@pytest.mark.parametrize(
    "content",
    [
        "---\n[not, a, mapping]\n---\nbody",
        "---\nupdated: [broken\n---\nbody",
        "---\nupdated: value\nbody",
        "\ufeff---\nupdated: value\n---\nbody",
    ],
)
def test_unsafe_frontmatter_raises(content):
    with pytest.raises(FrontmatterError):
        parse_frontmatter(content)
