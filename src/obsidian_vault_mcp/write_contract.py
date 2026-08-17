"""Server-side write contract gate.

Moves mechanically enforceable ``_SCHEMA.md`` invariants into the shared write
path (``vault.write_file_atomic`` / ``vault.move_path`` / ``vault.delete_path``)
so every mutation tool is covered by construction rather than by convention.

Feature mode is a single environment variable, ``VAULT_WRITE_CONTRACT_MODE``:

  - ``off``     — gate does not run at all (zero overhead, current behaviour).
  - ``shadow``  — every registered rule runs and issues are logged, but no
                  write is ever blocked. This is the default.
  - ``enforce`` — rules explicitly marked ``enforced=True`` (i.e. proven safe
                  against a shadow scan of the real vaults) can block a write.
                  Rules not marked ``enforced`` keep logging in shadow mode
                  even while the global mode is ``enforce`` — an uncertain
                  rule never blocks just because the master switch flipped.

A blocked write must never touch the filesystem: ``evaluate()`` is called
before any temp file is created, so a rejection leaves the source file's
bytes (and mtime) untouched.

Enforcement status (as of the vault-write-contract-gate build, based on
scripts/write_contract_shadow_scan.py against all three real vaults --
2112 bs-brain + 82 alcove-brain + 767 cb-brain = 2961 files, 0 operational
errors):

  - ``unsafe-path-chars`` (enforced) -- 0 hits across all 2961 files.
  - ``protected-structural-file`` (enforced) -- 4 hits, all literally named
    _SCHEMA.md (the real schema doc in each of the 3 vaults, plus one
    template copy under an alcove-brain-seed/ folder) -- exactly the
    intended targets, not false positives.
  - ``frontmatter-parseable`` (shadow-only) -- 1 hit: a legitimate file
    (Personal/Build Orchestrator/orchestrator-backlog.md in bs-brain) whose
    first line is a markdown thematic-break `---` with no actual
    frontmatter, which this rule cannot yet distinguish from an unterminated
    frontmatter block. Known limitation; do not enforce until fixed.
  - ``frontmatter-dates-quoted`` (advisory severity; shadow-only) -- 252
    hits across the 3 vaults. Bare unquoted dates are widespread in
    existing content; cannot be enforced without a separate remediation
    pass.
  - ``unsafe-file-extension`` (advisory severity; shadow-only) -- 120 hits,
    all legitimate non-.md infrastructure files (.py, .yaml, .gitignore,
    .tsv). Allowlist intentionally not tightened further.
  - ``frontmatter-required-fields`` (advisory severity; shadow-only) -- 0
    hits; no vault has opted in via _schema_rules.json yet.
  - ``protected-read-policy-full-rewrite`` (shadow-only) -- a static scan
    can only test the true no-op case (0 hits there by construction, see
    the rule's own docstring); it has not been proven against real editing
    traffic. Recommended follow-up: promote to enforced after a shadow-
    observation period once real vault_write/vault_batch_write traffic
    against read_policy: section-only files has been reviewed.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable

from .frontmatter_safe import FrontmatterError, parse_frontmatter

logger = logging.getLogger(__name__)


class WriteContractError(ValueError):
    """Raised when an enforced rule rejects a proposed write.

    Subclasses ValueError so every existing tool's `except ValueError as e`
    handler reports it as a normal, structured error with no code changes.
    """


# --------------------------------------------------------------------------
# Core types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    rule_id: str
    severity: str  # "reject" (mechanically safe to enforce) or "advisory"
    message: str


@dataclass(frozen=True)
class WriteContext:
    """Everything a content-mutation validator needs about one proposed write."""

    path: str  # vault-relative path, already resolved/validated by resolve_vault_path
    old_content: str | None  # None if the file does not yet exist
    new_content: str
    tool: str = ""  # calling tool name, for logging/diagnostics only


@dataclass(frozen=True)
class PathMutationContext:
    """Everything a path-mutation (move/delete) validator needs."""

    path: str
    operation: str  # "move" | "delete"
    destination: str | None = None


Validator = Callable[[WriteContext], list[ValidationIssue]]
PathValidator = Callable[[PathMutationContext], list[ValidationIssue]]

_CONTENT_REGISTRY: dict[str, Validator] = {}
_PATH_REGISTRY: dict[str, PathValidator] = {}
_ENFORCED_RULES: set[str] = set()


def register_content_rule(rule_id: str, *, enforced: bool = False) -> Callable[[Validator], Validator]:
    def deco(fn: Validator) -> Validator:
        _CONTENT_REGISTRY[rule_id] = fn
        if enforced:
            _ENFORCED_RULES.add(rule_id)
        return fn

    return deco


def register_path_rule(rule_id: str, *, enforced: bool = False) -> Callable[[PathValidator], PathValidator]:
    def deco(fn: PathValidator) -> PathValidator:
        _PATH_REGISTRY[rule_id] = fn
        if enforced:
            _ENFORCED_RULES.add(rule_id)
        return fn

    return deco


def is_rule_enforced(rule_id: str) -> bool:
    return rule_id in _ENFORCED_RULES


def registered_content_rules() -> list[str]:
    return sorted(_CONTENT_REGISTRY)


def registered_path_rules() -> list[str]:
    return sorted(_PATH_REGISTRY)


def get_mode() -> str:
    mode = os.environ.get("VAULT_WRITE_CONTRACT_MODE", "shadow").strip().lower()
    if mode not in ("off", "shadow", "enforce"):
        return "shadow"
    return mode


@dataclass(frozen=True)
class GateResult:
    mode: str
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def blocking_issues(self) -> list[ValidationIssue]:
        if self.mode != "enforce":
            return []
        return [i for i in self.issues if i.severity == "reject" and is_rule_enforced(i.rule_id)]

    @property
    def blocked(self) -> bool:
        return bool(self.blocking_issues)


def _run(registry: dict[str, Callable], ctx) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for rule_id, fn in registry.items():
        try:
            issues.extend(fn(ctx))
        except Exception:  # a validator bug must never break or block a write
            logger.exception("write-contract validator %r raised unexpectedly", rule_id)
    return issues


def evaluate_content(ctx: WriteContext) -> GateResult:
    mode = get_mode()
    if mode == "off":
        return GateResult(mode=mode, issues=[])
    issues = _run(_CONTENT_REGISTRY, ctx)
    _log_issues(mode, ctx.path, ctx.tool, issues)
    return GateResult(mode=mode, issues=issues)


def evaluate_path_mutation(ctx: PathMutationContext) -> GateResult:
    mode = get_mode()
    if mode == "off":
        return GateResult(mode=mode, issues=[])
    issues = _run(_PATH_REGISTRY, ctx)
    _log_issues(mode, ctx.path, ctx.operation, issues)
    return GateResult(mode=mode, issues=issues)


def _log_issues(mode: str, path: str, tool: str, issues: list[ValidationIssue]) -> None:
    for issue in issues:
        would_block = mode == "enforce" and issue.severity == "reject" and is_rule_enforced(issue.rule_id)
        logger.warning(
            "write-contract[%s] rule=%s severity=%s enforced=%s blocked=%s tool=%s path=%s: %s",
            mode,
            issue.rule_id,
            issue.severity,
            is_rule_enforced(issue.rule_id),
            would_block,
            tool,
            path,
            issue.message,
        )


def enforce(ctx: WriteContext) -> GateResult:
    """Evaluate content rules and raise WriteContractError if any are blocking."""
    result = evaluate_content(ctx)
    if result.blocked:
        summary = "; ".join(f"{i.rule_id}: {i.message}" for i in result.blocking_issues)
        raise WriteContractError(f"write rejected by write-contract gate: {summary}")
    return result


def enforce_path_mutation(ctx: PathMutationContext) -> GateResult:
    result = evaluate_path_mutation(ctx)
    if result.blocked:
        summary = "; ".join(f"{i.rule_id}: {i.message}" for i in result.blocking_issues)
        raise WriteContractError(f"{ctx.operation} rejected by write-contract gate: {summary}")
    return result


# --------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------

ISO_DATEISH_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[Tt ][0-9:.+-]+(?:[Zz]|[+-]\d{2}:?\d{2})?)?$"
)

# Characters that are illegal in a filename on Windows (this project's live
# checkout runs from a Windows-mounted filesystem: /mnt/c/Users/.../).
_WINDOWS_RESERVED_CHARS_RE = re.compile(r'[<>:"|?*\x00-\x1f]')
_WINDOWS_RESERVED_BASENAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


@register_content_rule("frontmatter-parseable", enforced=False)
def _validate_frontmatter_parseable(ctx: WriteContext) -> list[ValidationIssue]:
    """The proposed new content's leading frontmatter block, if any, must parse.

    Catches: unterminated blocks, non-mapping top level, duplicate keys, a
    leading UTF-8 BOM, and any other YAML the strict round-trip parser rejects.
    """
    try:
        parse_frontmatter(ctx.new_content)
    except FrontmatterError as exc:
        return [ValidationIssue("frontmatter-parseable", "reject", str(exc))]
    return []


@register_content_rule("frontmatter-dates-quoted", enforced=False)
def _validate_dates_quoted(ctx: WriteContext) -> list[ValidationIssue]:
    """Date-ish frontmatter scalars must be quoted strings, not bare YAML dates.

    A bare ``updated: 2026-08-17`` round-trips through PyYAML/ruamel as an
    actual ``datetime.date`` object, which downstream tools then re-serialise
    inconsistently. This is the exact corruption class vault-write-path-
    safety-v2 fixed for the maintenance scripts; this rule catches it for the
    live MCP write tools too (e.g. vault_write's merge_frontmatter path,
    which still uses python-frontmatter and does not quote dates itself).
    """
    try:
        document = parse_frontmatter(ctx.new_content)
    except FrontmatterError:
        return []  # frontmatter-parseable already reports this
    if document is None:
        return []

    issues: list[ValidationIssue] = []

    def _walk(value, key_path: str) -> None:
        import datetime as _dt

        if isinstance(value, (_dt.datetime, _dt.date)):
            issues.append(
                ValidationIssue(
                    "frontmatter-dates-quoted",
                    "advisory",
                    f"field {key_path!r} is a bare YAML date/datetime; quote it as a string",
                )
            )
        elif isinstance(value, dict):
            for k, v in value.items():
                _walk(v, f"{key_path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _walk(v, f"{key_path}[{i}]")

    for field_name, field_value in document.metadata.items():
        _walk(field_value, str(field_name))

    return issues


# Tools that pass write_file_atomic the *entire* intended file body, so a
# read_policy: section-only marker is meaningless protection against them.
# Narrow-edit tools (patch_section/append/str_replace/batch_str_replace/
# batch_frontmatter_update) only ever touch the specific piece they name and
# are exempt by construction.
_WHOLE_FILE_REPLACE_TOOLS = {"vault_write", "vault_batch_write"}


@register_content_rule("protected-read-policy-full-rewrite", enforced=False)
def _validate_read_policy_full_rewrite(ctx: WriteContext) -> list[ValidationIssue]:
    """Files marked read_policy: section-only should not be wholesale-replaced.

    That frontmatter marker exists specifically because the file is large and
    curated; a full-body overwrite (vault_write, vault_batch_write) throws
    away everything not present in the caller's new content. Which tool was
    called is the mechanically reliable signal here -- text-diffing old vs.
    new body is not, since a legitimate vault_patch_section edit can change
    most of a section's text and still be the sanctioned narrow-edit path.
    """
    if ctx.tool not in _WHOLE_FILE_REPLACE_TOOLS:
        return []
    if ctx.old_content is None:
        return []
    try:
        old_document = parse_frontmatter(ctx.old_content)
    except FrontmatterError:
        return []
    if old_document is None:
        return []
    if old_document.metadata.get("read_policy") != "section-only":
        return []
    if ctx.old_content == ctx.new_content:
        return []

    return [
        ValidationIssue(
            "protected-read-policy-full-rewrite",
            "reject",
            "file has read_policy: section-only and this write replaces the whole body; "
            "use vault_patch_section, vault_append, or vault_str_replace instead",
        )
    ]


@register_content_rule("unsafe-path-chars", enforced=True)
def _validate_unsafe_path_chars(ctx: WriteContext) -> list[ValidationIssue]:
    """Reject paths containing characters/names illegal on Windows filesystems.

    The live checkout backing all three vault-mcp instances is served from a
    Windows-mounted path; a filename with e.g. a bare `:` or a trailing `.`
    can silently fail or get mangled at the filesystem layer well after this
    server's own path-traversal checks pass.
    """
    issues: list[ValidationIssue] = []
    bad_chars = sorted(set(_WINDOWS_RESERVED_CHARS_RE.findall(ctx.path)))
    if bad_chars:
        issues.append(
            ValidationIssue(
                "unsafe-path-chars",
                "reject",
                f"path contains characters illegal on Windows filesystems: {bad_chars!r}",
            )
        )
    for part in ctx.path.replace("\\", "/").split("/"):
        stem = part.rsplit(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED_BASENAMES:
            issues.append(
                ValidationIssue(
                    "unsafe-path-chars",
                    "reject",
                    f"path component {part!r} is a reserved Windows device name",
                )
            )
    return issues


_SAFE_EXTENSIONS = {
    ".md", ".canvas", ".base", ".json", ".csv", ".txt", ".excalidraw",
}


@register_content_rule("unsafe-file-extension", enforced=False)
def _validate_file_extension(ctx: WriteContext) -> list[ValidationIssue]:
    """Advisory: flag writes to extensions outside the known-safe vault content set.

    Kept advisory-only (never enforced) until a shadow scan of real vault
    traffic/content proves the allowlist is complete enough not to false-
    positive on legitimate attachment types.
    """
    name = ctx.path.rsplit("/", 1)[-1]
    if "." not in name:
        return []  # extensionless files (e.g. LICENSE-style) are out of scope
    ext = "." + name.rsplit(".", 1)[-1].lower()
    if ext in _SAFE_EXTENSIONS:
        return []
    return [
        ValidationIssue(
            "unsafe-file-extension",
            "advisory",
            f"writing file with extension {ext!r}, outside the known vault content allowlist {sorted(_SAFE_EXTENSIONS)}",
        )
    ]


@register_content_rule("frontmatter-required-fields", enforced=False)
def _validate_required_fields(ctx: WriteContext) -> list[ValidationIssue]:
    """Advisory: required-fields-per-type, sourced from an optional, per-vault
    ``_schema_rules.json`` at the vault root.

    Deliberately NOT hardcoded to any one vault's conventions: Alcove Brain
    and CB Brain have no ``_SCHEMA.md`` today, so a rule baked from BS
    Brain's schema would be pure false-positive noise there. Absent the
    opt-in file, this rule never fires — safe by construction on all three
    deployed vaults until a vault's own maintainers opt in.
    """
    rules = _load_schema_rules()
    if not rules:
        return []
    try:
        document = parse_frontmatter(ctx.new_content)
    except FrontmatterError:
        return []
    if document is None:
        return []
    doc_type = document.metadata.get("type")
    if doc_type is None:
        return []
    required = rules.get(str(doc_type))
    if not required:
        return []
    missing = [f for f in required if f not in document.metadata]
    if not missing:
        return []
    return [
        ValidationIssue(
            "frontmatter-required-fields",
            "advisory",
            f"type {doc_type!r} is missing required field(s): {missing}",
        )
    ]


_schema_rules_cache: dict | None = None
_schema_rules_cache_mtime: float | None = None


def _load_schema_rules() -> dict:
    """Load {type: [required_field, ...]} from <vault_root>/_schema_rules.json.

    Returns {} if the file is absent or malformed (never raises -- this rule
    is opt-in and must never break writes because of a typo in the JSON).
    """
    global _schema_rules_cache, _schema_rules_cache_mtime
    from . import config
    import json as _json

    rules_path = config.VAULT_PATH / "_schema_rules.json"
    try:
        mtime = rules_path.stat().st_mtime
    except OSError:
        _schema_rules_cache = None
        _schema_rules_cache_mtime = None
        return {}

    if _schema_rules_cache is not None and _schema_rules_cache_mtime == mtime:
        return _schema_rules_cache

    try:
        data = _json.loads(rules_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("_schema_rules.json must be a JSON object")
    except Exception:
        logger.exception("failed to load %s; frontmatter-required-fields rule disabled", rules_path)
        _schema_rules_cache = {}
        _schema_rules_cache_mtime = mtime
        return {}

    _schema_rules_cache = data
    _schema_rules_cache_mtime = mtime
    return data


import posixpath

# Matched by basename, not full path: _SCHEMA.md lives at e.g.
# "BS 2nd Brain/_SCHEMA.md" in the real vault, not at the vault root. A shadow
# scan against the real vaults caught an earlier, exact-full-path version of
# this rule silently never matching the real file -- see the write-contract-
# gate build log for the details.
_PROTECTED_STRUCTURAL_BASENAMES = {"_SCHEMA.md"}


@register_path_rule("protected-structural-file", enforced=True)
def _validate_protected_root_file(ctx: PathMutationContext) -> list[ValidationIssue]:
    """Block move/delete of a short list of critical structural files, by basename."""
    if posixpath.basename(ctx.path) in _PROTECTED_STRUCTURAL_BASENAMES:
        return [
            ValidationIssue(
                "protected-structural-file",
                "reject",
                f"{ctx.path!r} is a protected structural file and cannot be {ctx.operation}d",
            )
        ]
    return []
