"""canonical_state.py -- deterministic reader/validator for canonical-state records
(build: vault-canonical-state-model).

Problem this closes: the vault's only durable "is this still true" signal was
file-write metadata (frontmatter `updated`, or mtime) on prose documents
(infrastructure.md, hot.md). Neither field distinguishes "the fact changed" from
"someone touched the file" (a curation pass, a reflow, a typo fix), so nothing
in the vault could answer "what is the CURRENT state of component X" without a
human (or an LLM) re-reading prose and guessing. This module adds one narrow,
additive authority for that one question -- not a general graph database, not a
replacement for prose.

Canonical-state records are individual Markdown files, one per component, each
with a frontmatter block validated by this module. See the schema doc
(BS 2nd Brain/Alcove/Infrastructure/Canonical State/README.md) for the field
reference and the full precedence rule; the short version:

    canonical-state (this)  >  changelog  >  SYSTEM-FACTS.md  >  hot.md

for CURRENT-STATE questions specifically. Changelog remains the historical
record of record; SYSTEM-FACTS.md remains the place for corrections/lessons;
hot.md remains an ephemeral, budget-capped session cache. None of those three
are superseded or migrated by this build -- see the spec's explicit "do not
bulk migrate" instruction.

Everything here is read-only and zero-LLM: no semantic search, no embeddings,
just frontmatter parsing (via ``frontmatter_safe.parse_frontmatter``, already
strict about malformed YAML) plus deterministic string/date comparisons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .frontmatter_safe import FrontmatterError, parse_frontmatter

RECORD_TYPE = "canonical-state"

REQUIRED_FIELDS = ("type", "component_id", "state", "content_updated", "verified_at", "source")
OPTIONAL_FIELDS = ("status_changed_at", "superseded_by", "supersedes", "updated")

_DATE_FIELDS = ("content_updated", "verified_at", "status_changed_at")

_COMPONENT_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class CanonicalStateError(ValueError):
    """Raised for a structurally invalid canonical-state record."""


@dataclass(frozen=True)
class CanonicalStateRecord:
    """One parsed, individually-valid canonical-state record.

    ``path`` is the record's location; ``component_id``/``state`` are the two
    fields every reader needs first. ``is_current`` is true iff the record
    carries no ``superseded_by`` -- superseded records stay on disk as
    history, they just drop out of ``resolve_current_state``.
    """

    path: Path
    component_id: str
    state: str
    content_updated: str
    verified_at: str
    source: str
    status_changed_at: str | None
    superseded_by: str | None
    supersedes: str | None

    @property
    def is_current(self) -> bool:
        return not self.superseded_by


class DuplicateAuthorityError(CanonicalStateError):
    """Raised when more than one current record claims the same component_id."""

    def __init__(self, component_id: str, records: list[CanonicalStateRecord]):
        self.component_id = component_id
        self.records = records
        paths = ", ".join(str(r.path) for r in records)
        super().__init__(
            f"duplicate canonical authority for component_id {component_id!r}: "
            f"{len(records)} current records ({paths})"
        )


def _is_dateish(value: object) -> bool:
    if isinstance(value, (date, datetime)):
        return True
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _as_str(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def validate_frontmatter(metadata: dict, *, expected_component_id: str | None = None) -> list[str]:
    """Return a list of validation error strings; empty means valid.

    Pure function over an already-parsed frontmatter mapping -- no file I/O,
    so it is reusable against both on-disk records and not-yet-written drafts.
    ``expected_component_id`` (typically the filename stem) catches the
    derived-view conflict where the frontmatter's declared id and the id a
    reader would derive from the record's own location disagree.
    """
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in metadata or metadata.get(field) in (None, ""):
            errors.append(f"missing required field {field!r}")

    if metadata.get("type") not in (None, "", RECORD_TYPE):
        errors.append(f"type must be {RECORD_TYPE!r}, got {metadata.get('type')!r}")

    component_id = metadata.get("component_id")
    if isinstance(component_id, str) and component_id and not _COMPONENT_ID_RE.match(component_id):
        errors.append(
            f"component_id {component_id!r} must be lowercase kebab-case "
            "(matching ^[a-z0-9]+(-[a-z0-9]+)*$)"
        )
    if (
        expected_component_id is not None
        and isinstance(component_id, str)
        and component_id
        and component_id != expected_component_id
    ):
        errors.append(
            f"derived-view conflict: frontmatter component_id {component_id!r} does not "
            f"match the record's filename-derived id {expected_component_id!r}"
        )

    state = metadata.get("state")
    if state is not None and (not isinstance(state, str) or not state.strip()):
        errors.append("state must be a non-empty string")

    for field in _DATE_FIELDS:
        value = metadata.get(field)
        if value is not None and not _is_dateish(value):
            errors.append(f"{field} must be an ISO date/datetime, got {value!r}")

    content_updated = metadata.get("content_updated")
    verified_at = metadata.get("verified_at")
    if _is_dateish(content_updated) and _is_dateish(verified_at):
        cu = datetime.fromisoformat(_as_str(content_updated).replace("Z", "+00:00"))
        va = datetime.fromisoformat(_as_str(verified_at).replace("Z", "+00:00"))
        if va < cu:
            errors.append(
                "verified_at predates content_updated -- a fact cannot be verified "
                "before it was true"
            )

    source = metadata.get("source")
    if source is not None and (not isinstance(source, str) or not source.strip()):
        errors.append("source must be a non-empty string")

    superseded_by = metadata.get("superseded_by")
    if superseded_by is not None and (not isinstance(superseded_by, str) or not superseded_by.strip()):
        errors.append("superseded_by must be a non-empty string when present")

    return errors


def _filename_derived_component_id(path: Path) -> str:
    """The id a reader would derive from a record's own location.

    Multiple files legitimately share one ``component_id`` (a supersession
    chain, or -- deliberately -- a duplicate-authority bug this module exists
    to catch), so the filename stem cannot simply equal ``component_id`` in
    general. Convention: ``<component_id>.md`` or ``<component_id>--<variant>.md``,
    ``--`` chosen because it cannot appear inside a valid kebab-case id
    (``_COMPONENT_ID_RE`` forbids consecutive hyphens), so splitting on it is
    unambiguous.
    """
    return path.stem.split("--", 1)[0]


def parse_record(path: Path) -> tuple[CanonicalStateRecord | None, list[str]]:
    """Parse and validate one record file. Never raises on a malformed record
    -- returns ``(None, errors)`` instead, so a caller scanning a whole
    directory can keep going past one bad file."""
    path = Path(path)
    expected_id = _filename_derived_component_id(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [f"could not read {path}: {exc}"]

    try:
        document = parse_frontmatter(text)
    except FrontmatterError as exc:
        return None, [f"{path}: {exc}"]
    if document is None:
        return None, [f"{path}: no frontmatter block"]

    metadata = dict(document.metadata)
    errors = validate_frontmatter(metadata, expected_component_id=expected_id)
    if errors:
        return None, [f"{path}: {e}" for e in errors]

    record = CanonicalStateRecord(
        path=path,
        component_id=str(metadata["component_id"]),
        state=str(metadata["state"]),
        content_updated=_as_str(metadata["content_updated"]),
        verified_at=_as_str(metadata["verified_at"]),
        source=str(metadata["source"]),
        status_changed_at=_as_str(metadata["status_changed_at"]) if metadata.get("status_changed_at") else None,
        superseded_by=str(metadata["superseded_by"]) if metadata.get("superseded_by") else None,
        supersedes=str(metadata["supersedes"]) if metadata.get("supersedes") else None,
    )
    return record, []


def load_all_records(records_dir: Path) -> tuple[list[CanonicalStateRecord], list[str]]:
    """Parse every ``*.md`` file directly under ``records_dir``.

    Not recursive by design -- canonical-state records are a flat, additive
    namespace, not a tree to walk. Returns ``(valid_records, errors)``; a
    malformed record contributes to ``errors`` and is otherwise skipped.
    """
    records_dir = Path(records_dir)
    records: list[CanonicalStateRecord] = []
    errors: list[str] = []
    if not records_dir.is_dir():
        return records, errors
    for path in sorted(records_dir.glob("*.md")):
        record, record_errors = parse_record(path)
        if record is not None:
            records.append(record)
        errors.extend(record_errors)
    return records, errors


def resolve_current_state(component_id: str, records_dir: Path) -> CanonicalStateRecord | None:
    """Deterministically resolve the current record for one component_id.

    No semantic search, no fuzzy matching -- exact id match only, filtered to
    records with no ``superseded_by``. Returns ``None`` if no current record
    exists. Raises ``DuplicateAuthorityError`` if more than one current record
    claims the same id -- callers should treat that as a data-integrity bug to
    fix in the vault, not as an ambiguity to silently pick a winner for.
    """
    records, _errors = load_all_records(records_dir)
    matches = [r for r in records if r.component_id == component_id]
    current = [r for r in matches if r.is_current]
    if len(current) > 1:
        raise DuplicateAuthorityError(component_id, current)
    return current[0] if current else None


def scan_duplicate_authority(records_dir: Path) -> dict[str, list[CanonicalStateRecord]]:
    """Report-only duplicate-authority scan across every record in ``records_dir``.

    Returns a mapping of component_id -> list of current records, restricted
    to component_ids with more than one current record. Never raises, never
    writes -- see ``scripts/canonical_state_scan.py`` for the CLI wrapper.
    """
    records, _errors = load_all_records(records_dir)
    by_component: dict[str, list[CanonicalStateRecord]] = {}
    for record in records:
        if record.is_current:
            by_component.setdefault(record.component_id, []).append(record)
    return {cid: recs for cid, recs in by_component.items() if len(recs) > 1}
