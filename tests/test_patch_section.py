"""Tests for vault_patch_section -- single-section replacement without duplicate headings.

Regression coverage for a bug where the tool always kept the file's original
heading line and then blindly concatenated the caller's `content` after it. If
`content` itself started with a heading (matching or not), the result had two
heading lines instead of one. See BS Brain build log
vault-patch-section-heading-fix-output.md for the incident writeup.
"""

import json

import pytest

from obsidian_vault_mcp.tools.write import vault_patch_section


MULTI_SECTION = (
    "---\n"
    "title: Fixture\n"
    "status: active\n"
    "---\n"
    "## Intro\n"
    "Intro content here.\n"
    "\n"
    "## Foo\n"
    "old foo line1\n"
    "old foo line2\n"
    "\n"
    "### Sub\n"
    "sub content\n"
    "\n"
    "## Bar\n"
    "bar content\n"
)


def _write(vault_dir, name, text):
    path = vault_dir / name
    path.write_text(text)
    return path


def _count_heading_occurrences(text, heading):
    return sum(1 for line in text.splitlines() if line == heading)


def test_content_without_heading_yields_single_heading(vault_dir):
    _write(vault_dir, "note.md", MULTI_SECTION)
    result = json.loads(vault_patch_section("note.md", "## Foo", "new foo body"))
    assert "error" not in result
    text = (vault_dir / "note.md").read_text()
    assert _count_heading_occurrences(text, "## Foo") == 1
    assert "new foo body" in text


def test_content_with_matching_heading_not_duplicated(vault_dir):
    _write(vault_dir, "note.md", MULTI_SECTION)
    result = json.loads(vault_patch_section("note.md", "## Foo", "## Foo\nnew foo body"))
    assert "error" not in result
    text = (vault_dir / "note.md").read_text()
    assert _count_heading_occurrences(text, "## Foo") == 1
    assert "new foo body" in text


def test_content_with_different_heading_raises_and_writes_nothing(vault_dir):
    _write(vault_dir, "note.md", MULTI_SECTION)
    before = (vault_dir / "note.md").read_text()
    result = json.loads(
        vault_patch_section("note.md", "## Foo", "### Different Heading\nsome content")
    )
    assert "error" in result
    after = (vault_dir / "note.md").read_text()
    assert after == before


def test_heading_level_two_preserved(vault_dir):
    _write(vault_dir, "note.md", MULTI_SECTION)
    vault_patch_section("note.md", "## Foo", "## Foo\nreplaced")
    text = (vault_dir / "note.md").read_text()
    assert "## Foo\nreplaced" in text
    assert _count_heading_occurrences(text, "## Foo") == 1


def test_heading_level_three_preserved(vault_dir):
    _write(vault_dir, "note.md", MULTI_SECTION)
    vault_patch_section("note.md", "### Sub", "### Sub\nreplaced sub")
    text = (vault_dir / "note.md").read_text()
    assert "### Sub\nreplaced sub" in text
    assert _count_heading_occurrences(text, "### Sub") == 1


def test_section_first_in_file(vault_dir):
    _write(vault_dir, "note.md", MULTI_SECTION)
    vault_patch_section("note.md", "## Intro", "new intro")
    text = (vault_dir / "note.md").read_text()
    assert _count_heading_occurrences(text, "## Intro") == 1
    assert "new intro" in text
    # frontmatter still precedes it
    assert text.startswith("---\ntitle: Fixture\nstatus: active\n---\n## Intro\n")


def test_section_last_in_file(vault_dir):
    _write(vault_dir, "note.md", MULTI_SECTION)
    vault_patch_section("note.md", "## Bar", "## Bar\nnew bar content")
    text = (vault_dir / "note.md").read_text()
    assert _count_heading_occurrences(text, "## Bar") == 1
    assert text.rstrip("\n").endswith("new bar content")


def test_section_only_section_in_file(vault_dir):
    only = "---\ntitle: Solo\n---\n## Only\noriginal body\n"
    _write(vault_dir, "solo.md", only)
    vault_patch_section("solo.md", "## Only", "replaced body")
    text = (vault_dir / "solo.md").read_text()
    assert _count_heading_occurrences(text, "## Only") == 1
    assert text == "---\ntitle: Solo\n---\n## Only\nreplaced body\n"


def test_adjacent_sections_unchanged_byte_comparison(vault_dir):
    _write(vault_dir, "note.md", MULTI_SECTION)
    vault_patch_section("note.md", "## Foo", "changed foo body")
    text = (vault_dir / "note.md").read_text()

    intro_block = "## Intro\nIntro content here.\n\n"
    bar_block = "## Bar\nbar content\n"
    assert intro_block in text
    assert text.endswith(bar_block)


def test_frontmatter_unchanged_byte_comparison(vault_dir):
    _write(vault_dir, "note.md", MULTI_SECTION)
    vault_patch_section("note.md", "## Bar", "changed bar body")
    text = (vault_dir / "note.md").read_text()
    frontmatter = "---\ntitle: Fixture\nstatus: active\n---\n"
    assert text.startswith(frontmatter)


def test_empty_content_empties_section_but_keeps_heading(vault_dir):
    # "## Foo" contains "### Sub" as a nested subsection, so emptying "## Foo"
    # legitimately removes "### Sub" along with it -- only the next same-or-higher
    # level heading ("## Bar") survives as the section boundary.
    _write(vault_dir, "note.md", MULTI_SECTION)
    result = json.loads(vault_patch_section("note.md", "## Foo", ""))
    assert "error" not in result
    text = (vault_dir / "note.md").read_text()
    assert _count_heading_occurrences(text, "## Foo") == 1
    assert "## Foo\n## Bar" in text


@pytest.mark.parametrize("supplied_content", ["new foo body", "## Foo\nnew foo body"])
def test_idempotency_double_apply_matches_single_apply(vault_dir, supplied_content):
    _write(vault_dir, "note.md", MULTI_SECTION)
    vault_patch_section("note.md", "## Foo", supplied_content)
    once = (vault_dir / "note.md").read_text()

    vault_patch_section("note.md", "## Foo", supplied_content)
    twice = (vault_dir / "note.md").read_text()

    assert once == twice
    assert _count_heading_occurrences(twice, "## Foo") == 1
