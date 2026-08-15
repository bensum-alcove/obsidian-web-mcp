"""Strict, conservative YAML frontmatter parsing and field updates."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.error import YAMLError
from ruamel.yaml.scalarstring import SingleQuotedScalarString


class FrontmatterError(ValueError):
    """Raised when a leading frontmatter block cannot be safely parsed."""


ISO_DATEISH_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[Tt ][0-9:.+-]+(?:[Zz]|[+-]\d{2}:?\d{2})?)?$"
)


def _yaml() -> YAML:
    parser = YAML(typ="rt")
    parser.allow_duplicate_keys = False
    parser.preserve_quotes = True
    return parser


@dataclass(frozen=True)
class FrontmatterDocument:
    metadata: CommentedMap
    body: str
    newline: str
    closing_newline: str


def parse_frontmatter(content: str) -> FrontmatterDocument | None:
    """Parse a strictly line-delimited leading frontmatter block.

    A missing leading block returns ``None``. A malformed block raises so callers
    cannot accidentally rewrite content that was not parsed successfully.
    """
    if content.startswith("\ufeff"):
        raise FrontmatterError("UTF-8 BOM before frontmatter is not supported")
    opening = re.match(r"\A---(\r\n|\n)", content)
    if not opening:
        return None

    newline = opening.group(1)
    position = opening.end()
    closing_start = None
    closing_end = None
    closing_newline = ""
    for line in content[position:].splitlines(keepends=True):
        line_without_eol = line.rstrip("\r\n")
        if line_without_eol == "---":
            closing_start = position
            closing_end = position + len(line)
            closing_newline = line[len(line_without_eol):]
            break
        position += len(line)

    if closing_start is None:
        raise FrontmatterError("unterminated frontmatter block")

    yaml_text = content[opening.end():closing_start]
    try:
        metadata = _yaml().load(yaml_text)
    except YAMLError as exc:
        raise FrontmatterError(f"unparseable frontmatter: {exc}") from exc
    if not isinstance(metadata, CommentedMap):
        raise FrontmatterError("frontmatter must be a YAML mapping")

    return FrontmatterDocument(
        metadata=metadata,
        body=content[closing_end:],
        newline=newline,
        closing_newline=closing_newline,
    )


def _normalise_dates(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return SingleQuotedScalarString(value.isoformat())
    if isinstance(value, str) and ISO_DATEISH_RE.fullmatch(value):
        return SingleQuotedScalarString(value)
    if isinstance(value, dict):
        for key in list(value):
            value[key] = _normalise_dates(value[key])
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _normalise_dates(item)
        return value
    if isinstance(value, tuple):
        return [_normalise_dates(item) for item in value]
    return value


def update_frontmatter_field(
    content: str, field: str, value: Any, *, require_existing: bool = True
) -> str:
    """Parse, update, and serialize one field while preserving body bytes/text."""
    document = parse_frontmatter(content)
    if document is None:
        raise FrontmatterError("no frontmatter block")
    if require_existing and field not in document.metadata:
        raise FrontmatterError(f"frontmatter field {field!r} is missing")

    normalised_value = _normalise_dates(value)
    current_value = _normalise_dates(deepcopy(document.metadata.get(field)))
    if str(current_value) == str(normalised_value):
        return content

    metadata = _normalise_dates(deepcopy(document.metadata))
    metadata[field] = normalised_value
    stream = StringIO()
    _yaml().dump(metadata, stream)
    dumped = stream.getvalue()
    if document.newline != "\n":
        dumped = dumped.replace("\n", document.newline)
    return (
        f"---{document.newline}{dumped}---{document.closing_newline}"
        f"{document.body}"
    )
