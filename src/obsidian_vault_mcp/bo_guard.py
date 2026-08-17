"""Build Orchestrator path-mutation guard (vault-bo-authoring-mcp-v1) -- SHADOW ONLY.

Hooks every Vault MCP mutation primitive that can change BO authority --
write/append/patch/frontmatter-update (all of which converge on
vault.write_file_atomic, per the vault-write-contract-gate build's inventory)
and move/rename/delete (vault.move_path / vault.delete_path) -- for two paths:

  - Personal/Build Orchestrator/specs/
  - Personal/Build Orchestrator/schedules/

This is a second, independently-flagged gate alongside write_contract.py's
general-purpose one: mode is BO_PATH_GUARD_MODE ("off" | "shadow" (default) |
"enforce"), read fresh on every call, never shared with VAULT_WRITE_CONTRACT_MODE.
vault-bo-authoring-mcp-v1 deploys this as shadow only -- every rule below is
"reject" severity (mechanically meaningful, not merely advisory) and fully
wired for enforce, but turning the mode to "enforce" is an explicit, separate,
later build (vault-bo-authoring-enforce-v1) gated on independent Codex review.

Validation logic is delegated entirely to the BO authoring contract adapter
(bo_contract.py -> authoring_contract.py's JSON CLI) for anything that needs
live BO state or schema knowledge -- this module never re-encodes BO's own
project/tier/risk-domain enums or its terminal/dispatched status semantics.
The one necessarily-local piece of domain knowledge is the small "which spec
statuses are still freely editable by casual authoring" allowlist below (specs/
have no dedicated CLI preflight op the way schedules do); it is deliberately
narrow (an allowlist of pre-dispatch statuses, not a mirror of BO's full status
taxonomy) and, being shadow-only in this build, a staleness in that allowlist
can only ever produce a noisy or missing log line -- never a wrong block.

If the authoring-contract adapter itself is unavailable, every rule here
reports its own findings as "reject" severity (matching the tools' own
fail-closed policy) -- but since this guard is deployed shadow-only, that can
still never block a real write in this build; it only ever logs.
"""

from __future__ import annotations

import logging
import os
import posixpath
from dataclasses import dataclass, field

from . import bo_contract
from .write_contract import PathMutationContext, ValidationIssue, WriteContext

logger = logging.getLogger(__name__)


class BOGuardError(ValueError):
    """Raised when an enforced BO-guard rule rejects a mutation.

    Subclasses ValueError so every existing tool's `except ValueError as e`
    handler reports it as a normal, structured error with no code changes.
    Never raised while BO_PATH_GUARD_MODE is "shadow" or "off" (the only modes
    this build deploys) -- enforce is implemented and tested, not activated.
    """


SPECS_PREFIX = "Personal/Build Orchestrator/specs/"
SCHEDULES_PREFIX = "Personal/Build Orchestrator/schedules/"

# Statuses a spec is still safely, casually editable in -- i.e. it has never
# been claimed by a dispatch. Deliberately an allowlist of the small
# "pre-dispatch" surface (mirrors authoring_contract.KNOWN_STATUSES minus
# DISPATCHED_STATUSES/TERMINAL_STATUSES) rather than a mirror of BO's full
# status vocabulary -- see module docstring for why this one piece of
# domain knowledge is kept local instead of round-tripped through the adapter.
_SPEC_FREELY_MUTABLE_STATUSES = {"proposed", "pending", "ready"}


def _is_bo_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.startswith(SPECS_PREFIX) or normalized.startswith(SCHEDULES_PREFIX)


def get_mode() -> str:
    mode = os.environ.get("BO_PATH_GUARD_MODE", "shadow").strip().lower()
    if mode not in ("off", "shadow", "enforce"):
        return "shadow"
    return mode


@dataclass(frozen=True)
class BOGateResult:
    mode: str
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def blocking_issues(self) -> list[ValidationIssue]:
        if self.mode != "enforce":
            return []
        return [i for i in self.issues if i.severity == "reject"]

    @property
    def blocked(self) -> bool:
        return bool(self.blocking_issues)


def _log_issues(mode: str, path: str, tool_or_op: str, issues: list[ValidationIssue]) -> None:
    for issue in issues:
        would_block = mode == "enforce" and issue.severity == "reject"
        logger.warning(
            "bo-guard[%s] rule=%s severity=%s blocked=%s tool=%s path=%s: %s",
            mode, issue.rule_id, issue.severity, would_block, tool_or_op, path, issue.message,
        )


def _build_id_from_spec_path(path: str) -> str:
    name = posixpath.basename(path.replace("\\", "/"))
    return name[: -len(".md")] if name.endswith(".md") else name


def _errors_to_issues(rule_id: str, errors: list[dict]) -> list[ValidationIssue]:
    return [
        ValidationIssue(rule_id, "reject", f"[{err.get('code')}] {err.get('message')}")
        for err in errors
    ]


def parse_schedule_builds(content: str) -> list[dict] | None:
    """Extract the `builds:` list from a schedule file's body.

    Schedule files are frontmatter + a markdown heading + a `builds:` YAML
    list; the heading line is a harmless YAML comment (`#`), so the body
    parses directly as YAML -- the same shape authoring_contract's own
    render_schedule_document produces. Returns None if the body isn't
    parseable or has no `builds:` list (callers treat that as "cannot
    evaluate", not as "zero builds").
    """
    import frontmatter
    import yaml

    try:
        post = frontmatter.loads(content)
        body = yaml.safe_load(post.content) or {}
    except Exception:
        return None
    builds = body.get("builds") if isinstance(body, dict) else None
    return builds if isinstance(builds, list) else None


def _schedule_rewrite_issues(ctx: WriteContext) -> list[ValidationIssue]:
    if ctx.old_content is None or ctx.old_content == ctx.new_content:
        return []  # brand-new schedule file, or a no-op write -- nothing bound yet / nothing changing
    new_builds = parse_schedule_builds(ctx.new_content)
    if new_builds is None:
        return [ValidationIssue(
            "bo-guard-schedule-rewrite", "advisory",
            f"could not parse a builds: list from the proposed content for {ctx.path!r}; "
            "BO schedule-authority preflight skipped for this write",
        )]
    try:
        result = bo_contract.preflight_schedule_rewrite(ctx.path, new_builds, mode="compat_existing")
    except bo_contract.BOContractError as e:
        return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
    return _errors_to_issues("bo-guard-schedule-rewrite", result.get("errors", []))


def _spec_rewrite_issues(ctx: WriteContext) -> list[ValidationIssue]:
    if ctx.old_content is None or ctx.old_content == ctx.new_content:
        return []  # brand-new spec, or a no-op write
    build_id = _build_id_from_spec_path(ctx.path)
    try:
        result = bo_contract.preflight_ids([build_id])
    except bo_contract.BOContractError as e:
        return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
    status = (result.get("results") or {}).get(build_id)
    if status is not None and status not in _SPEC_FREELY_MUTABLE_STATUSES:
        return [ValidationIssue(
            "bo-guard-spec-rewrite", "reject",
            f"{build_id!r} has status {status!r} — its spec content is bound to a dispatched-or-later "
            "or closed build and cannot be rewritten by the normal authoring path",
        )]
    return []


def _content_issues(ctx: WriteContext) -> list[ValidationIssue]:
    normalized = ctx.path.replace("\\", "/")
    if normalized.startswith(SCHEDULES_PREFIX):
        return _schedule_rewrite_issues(ctx)
    if normalized.startswith(SPECS_PREFIX):
        return _spec_rewrite_issues(ctx)
    return []


def _path_mutation_issues(ctx: PathMutationContext) -> list[ValidationIssue]:
    normalized = ctx.path.replace("\\", "/")
    if normalized.startswith(SCHEDULES_PREFIX):
        try:
            result = bo_contract.preflight_schedule_move(ctx.path)
        except bo_contract.BOContractError as e:
            return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
        return _errors_to_issues("bo-guard-schedule-move", result.get("errors", []))
    if normalized.startswith(SPECS_PREFIX):
        build_id = _build_id_from_spec_path(ctx.path)
        try:
            result = bo_contract.preflight_ids([build_id])
        except bo_contract.BOContractError as e:
            return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
        status = (result.get("results") or {}).get(build_id)
        if status is not None and status not in _SPEC_FREELY_MUTABLE_STATUSES:
            return [ValidationIssue(
                "bo-guard-spec-move", "reject",
                f"{build_id!r} has status {status!r} — {ctx.operation}ing its spec risks orphaning a "
                "bound or closed build (schedule_unbound-style regression)",
            )]
    return []


def evaluate_content(ctx: WriteContext) -> BOGateResult:
    mode = get_mode()
    if mode == "off" or not _is_bo_path(ctx.path):
        return BOGateResult(mode=mode, issues=[])
    issues = _content_issues(ctx)
    _log_issues(mode, ctx.path, ctx.tool, issues)
    return BOGateResult(mode=mode, issues=issues)


def evaluate_path_mutation(ctx: PathMutationContext) -> BOGateResult:
    mode = get_mode()
    if mode == "off" or not _is_bo_path(ctx.path):
        return BOGateResult(mode=mode, issues=[])
    issues = _path_mutation_issues(ctx)
    _log_issues(mode, ctx.path, ctx.operation, issues)
    return BOGateResult(mode=mode, issues=issues)


def enforce(ctx: WriteContext) -> BOGateResult:
    """Evaluate content rules and raise BOGuardError if any are blocking."""
    result = evaluate_content(ctx)
    if result.blocked:
        summary = "; ".join(f"{i.rule_id}: {i.message}" for i in result.blocking_issues)
        raise BOGuardError(f"write rejected by BO path guard: {summary}")
    return result


def enforce_path_mutation(ctx: PathMutationContext) -> BOGateResult:
    result = evaluate_path_mutation(ctx)
    if result.blocked:
        summary = "; ".join(f"{i.rule_id}: {i.message}" for i in result.blocking_issues)
        raise BOGuardError(f"{ctx.operation} rejected by BO path guard: {summary}")
    return result
