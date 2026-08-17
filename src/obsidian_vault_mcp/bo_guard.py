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
Deployed shadow-only in both vault-bo-authoring-mcp-v1 and this remediation
build -- every rule below is "reject" severity (mechanically meaningful, not
merely advisory) and fully wired for enforce, but turning the mode to
"enforce" is an explicit, separate, later build gated on independent Codex
review (codex-review-bo-authoring-contract-v2).

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

codex-review-bo-authoring-contract-v1 (2026-08-17) found this first cut fails
open in several ways -- see B1 in that review. This build's fixes, all still
shadow-only:

  - brand-new schedules/specs are no longer exempt from validation (the old
    ``old_content is None -> return []`` short-circuit only skips a true
    same-content no-op now);
  - an unparseable/malformed schedule rewrite is "reject" severity, not
    "advisory" -- an enforce-mode future build can actually block it;
  - schedule rewrites are validated against the COMPLETE resulting graph
    (every entry that will exist in the file after the write, each paired
    with its on-disk spec content), not just bound-row preservation;
  - spec rewrites are validated against the spec's own content shape
    (frontmatter/identity/tier/risk fields), not just a status lookup;
  - path mutation now evaluates the DESTINATION too, so an inbound move from
    a non-BO path into specs/ or schedules/ is validated as if it were a
    fresh write there;
  - BO directories (specs/, schedules/) and their ancestors are protected
    recursively: moving/deleting the directory itself, or any ancestor of
    it, is evaluated against every schedule/spec file currently on disk
    underneath it -- not just a single exact-path DB lookup that silently
    matches nothing for a directory-shaped path.

codex-review-bo-authoring-contract-v2 (2026-08-18) found three further
bypasses (B7-B9), still shadow-only, all closed in
vault-integrity-and-bo-authority-remediation-v2:

  - B7: a pending/ready spec rewrite validated only the spec's own content
    shape, never the graph of any schedule it's already bound to -- an
    individually well-formed spec edit (e.g. changing its project) could
    invalidate its schedule's project/dependency authority. Spec rewrites
    now also resolve the referring schedule (if any) and revalidate its
    complete resulting graph using the proposed spec bytes.
  - B8: destination validation only ran when the source was NOT also a BO
    path, so a specs/ <-> schedules/ cross-move skipped destination-type
    validation entirely, and inbound directory moves skipped content
    validation outright. Destination authority is now evaluated
    independently of the source's own BO-path status, and inbound directory
    moves enumerate and validate every nested artifact that would land
    under the destination (fail-closed on anything unevaluable).
  - B9: authoring_contract.validate_schedule_move() exempted terminal-only
    schedules from move/delete protection -- a schedule containing only
    closed rows could be renamed or deleted outright. It now rejects moving
    or deleting a schedule bound to ANY row, terminal or not; closed history
    remains authority and continuation requires a new schedule/new id.
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

# Fallback only, used solely when the adapter itself is unavailable (see
# _freely_mutable_statuses below) -- the authoritative set is now sourced
# from the adapter's own op=version vocabulary
# (known_statuses - terminal_statuses - dispatched_statuses), not hardcoded
# here (codex-review-bo-authoring-contract-v1, B5: "the status classification
# should be owned by and returned from the contract adapter").
_SPEC_FREELY_MUTABLE_STATUSES_FALLBACK = {"proposed", "pending", "ready"}

_freely_mutable_statuses_cache: set | None = None


def _freely_mutable_statuses() -> set:
    """Cached (process-lifetime) lookup of the adapter's own
    freely-mutable-status vocabulary. A restart is required to pick up a
    change in the adapter's vocabulary, matching how EXPECTED_SCHEMA_VERSION/
    EXPECTED_CONTRACT_VERSION drift already requires a restart. On adapter
    failure, the fallback allowlist above is used for that one call without
    poisoning the cache -- being shadow-only, a staleness here can only ever
    produce a wrong log line, never a wrong block.
    """
    global _freely_mutable_statuses_cache
    if _freely_mutable_statuses_cache is not None:
        return _freely_mutable_statuses_cache
    try:
        _freely_mutable_statuses_cache = bo_contract.freely_mutable_statuses()
    except bo_contract.BOContractError:
        return _SPEC_FREELY_MUTABLE_STATUSES_FALLBACK
    return _freely_mutable_statuses_cache


def _is_bo_path(path: str) -> bool:
    """True if `path` is inside specs/schedules, IS one of those directories,
    or is an ancestor of either -- e.g. moving/deleting
    "Personal/Build Orchestrator" (which contains both) must not silently
    skip evaluation just because it isn't itself under a specs/schedules
    prefix (codex-review-bo-authoring-contract-v1, B1)."""
    normalized = path.replace("\\", "/").rstrip("/")
    for prefix in (SPECS_PREFIX, SCHEDULES_PREFIX):
        prefix_no_slash = prefix.rstrip("/")
        if normalized.startswith(prefix) or (prefix_no_slash + "/").startswith(normalized + "/"):
            return True
    return False


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


def _schedule_project_from_content(content: str) -> str | None:
    import frontmatter
    try:
        return frontmatter.loads(content).metadata.get("project")
    except Exception:
        return None


def _read_vault_text(relative_path: str) -> str | None:
    """Best-effort read of a vault-relative path's raw text off disk.

    Used only for read-only shadow evaluation (never for a mutation) --
    returns None rather than raising for a missing/unreadable/binary file so
    a dangling spec_path reference degrades to "no content to check" instead
    of blowing up the guard.
    """
    from . import config

    try:
        full = (config.VAULT_PATH / relative_path).resolve()
        if not full.is_file():
            return None
        return full.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _files_under(relative_dir: str) -> list[str]:
    """Vault-relative file paths currently on disk under `relative_dir`.

    Used when a guard check must cover an entire directory being moved or
    deleted in bulk (e.g. the whole schedules/ directory, or an ancestor of
    it) rather than one exact path -- a single-path DB/adapter lookup
    silently matches nothing for a directory-shaped path.
    """
    from . import config

    base = (config.VAULT_PATH / relative_dir).resolve()
    vault_root = config.VAULT_PATH.resolve()
    if not base.is_dir():
        return []
    out: list[str] = []
    try:
        for p in sorted(base.rglob("*")):
            if p.is_file():
                try:
                    out.append(str(p.relative_to(vault_root)).replace("\\", "/"))
                except ValueError:
                    continue
    except OSError:
        return out
    return out


def _affected_paths(path: str, prefix: str) -> list[str]:
    """Vault-relative file paths that a move/delete of `path` would affect,
    given a protected directory `prefix` (SPECS_PREFIX or SCHEDULES_PREFIX).

    Three cases: `path` is a file/subpath under `prefix` (returns that one
    path); `path` IS `prefix`'s directory itself, or an ancestor of it
    (returns every file currently under `prefix` on disk -- the bulk case);
    otherwise empty.
    """
    normalized = path.replace("\\", "/").rstrip("/")
    prefix_no_slash = prefix.rstrip("/")
    if (prefix_no_slash + "/").startswith(normalized + "/"):
        return _files_under(prefix)
    if normalized.startswith(prefix):
        return [normalized]
    return []


def _nodes_for_schedule_builds(builds: list[dict], schedule_path: str, schedule_project: str | None) -> list[dict]:
    nodes = []
    for entry in builds:
        if not isinstance(entry, dict):
            continue
        spec_path = entry.get("spec_path")
        nodes.append({
            "build_id": entry.get("id"),
            "schedule_entry": entry,
            "spec_markdown": _read_vault_text(spec_path) or "" if isinstance(spec_path, str) else "",
            "schedule_path": schedule_path,
            "schedule_project": schedule_project,
        })
    return nodes


def _schedule_rewrite_issues(ctx: WriteContext) -> list[ValidationIssue]:
    if ctx.old_content is not None and ctx.old_content == ctx.new_content:
        return []  # true no-op write

    new_builds = parse_schedule_builds(ctx.new_content)
    if new_builds is None:
        # Previously "advisory" -- an unparseable/malformed rewrite is a real,
        # reject-severity finding so a future enforce-mode build can actually
        # block it (codex-review-bo-authoring-contract-v1, B1).
        return [ValidationIssue(
            "bo-guard-schedule-rewrite", "reject",
            f"proposed content for {ctx.path!r} does not parse as a valid schedule "
            "builds: list -- rejecting rather than allowing an unparseable rewrite through",
        )]

    issues: list[ValidationIssue] = []

    # 1. Bound-row preservation / terminal-entry-edit preflight. Runs
    #    regardless of whether this is a brand-new or existing schedule -- a
    #    brand-new schedule simply has no bound rows to preserve, which is a
    #    no-op result here, not an exemption from validation overall.
    try:
        preflight_result = bo_contract.preflight_schedule_rewrite(ctx.path, new_builds, mode="compat_existing")
    except bo_contract.BOContractError as e:
        return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
    issues.extend(_errors_to_issues("bo-guard-schedule-rewrite", preflight_result.get("errors", [])))

    # 2. Whole-resulting-graph validation: every entry that will exist in the
    #    file after this write, each paired with its on-disk spec content --
    #    not just the rows this particular write happens to touch. Closes the
    #    "mixed execution projects / duplicate un-ingested ID across the
    #    whole schedule not detected" gap (B2).
    schedule_project = _schedule_project_from_content(ctx.new_content)
    old_builds = parse_schedule_builds(ctx.old_content) if ctx.old_content else None
    old_by_id = {b.get("id"): b for b in (old_builds or []) if isinstance(b, dict)}
    new_ids = sorted({
        entry.get("id") for entry in new_builds
        if isinstance(entry, dict) and entry.get("id") is not None
        and entry != old_by_id.get(entry.get("id"))
    })
    nodes = _nodes_for_schedule_builds(new_builds, ctx.path, schedule_project)
    try:
        graph_result = bo_contract.validate_graph(nodes, mode="strict_new", new_ids=new_ids)
    except bo_contract.BOContractError as e:
        issues.append(ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e)))
        return issues
    issues.extend(_errors_to_issues("bo-guard-schedule-graph", graph_result.get("errors", [])))

    return issues


def _spec_rewrite_issues(ctx: WriteContext) -> list[ValidationIssue]:
    if ctx.old_content is not None and ctx.old_content == ctx.new_content:
        return []  # true no-op write

    build_id = _build_id_from_spec_path(ctx.path)
    try:
        result = bo_contract.preflight_ids([build_id])
    except bo_contract.BOContractError as e:
        return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
    status = (result.get("results") or {}).get(build_id)
    if status is not None and status not in _freely_mutable_statuses():
        return [ValidationIssue(
            "bo-guard-spec-rewrite", "reject",
            f"{build_id!r} has status {status!r} — its spec content is bound to a dispatched-or-later "
            "or closed build and cannot be rewritten by the normal authoring path",
        )]

    # Content-shape check (frontmatter validity, identity, tier, risk fields)
    # so a pre-dispatch spec can't be rewritten to arbitrary invalid bytes
    # just because its status is still freely-mutable (B1). Lenient
    # (compat_existing) since this may be a pre-v6 spec -- catches malformed
    # YAML / identity mismatch / unknown enums without false-positiving on
    # legacy risk-field shape.
    try:
        spec_result = bo_contract.preflight_spec_validate(
            build_id, ctx.new_content, ctx.path, mode="compat_existing",
        )
    except bo_contract.BOContractError as e:
        return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
    issues = _errors_to_issues("bo-guard-spec-rewrite", spec_result.get("errors", []))

    # Whole-graph validation of any schedule this build_id is already bound
    # to, using the PROPOSED spec bytes (not what's currently on disk) --
    # a spec that is individually well-formed can still invalidate its
    # referring schedule's project/dependency/graph authority (B7,
    # codex-review-bo-authoring-contract-v2). A spec not yet bound to any
    # schedule has nothing to revalidate here.
    issues.extend(_spec_rewrite_schedule_graph_issues(build_id, ctx))
    return issues


def _spec_rewrite_schedule_graph_issues(build_id: str, ctx: WriteContext) -> list[ValidationIssue]:
    try:
        sched_result = bo_contract.preflight_source_schedule([build_id])
    except bo_contract.BOContractError as e:
        return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
    source_schedule = (sched_result.get("results") or {}).get(build_id)
    if not source_schedule:
        return []  # not bound to any schedule yet -- nothing to revalidate

    schedule_content = _read_vault_text(source_schedule)
    if schedule_content is None:
        return []  # schedule file unreadable/missing -- degrade quietly, matching other _read_vault_text call sites

    builds = parse_schedule_builds(schedule_content)
    if builds is None:
        return []  # schedule itself unparseable -- its own rewrite path already guards this

    schedule_project = _schedule_project_from_content(schedule_content)
    nodes = _nodes_for_schedule_builds(builds, source_schedule, schedule_project)
    # Substitute the proposed new spec content for this build_id's node so
    # the graph is validated against the bytes that WOULD exist after this
    # write commits, not what's currently on disk under specs/.
    for node in nodes:
        if node["build_id"] == build_id:
            node["spec_markdown"] = ctx.new_content

    try:
        graph_result = bo_contract.validate_graph(nodes, mode="strict_new", new_ids=[build_id])
    except bo_contract.BOContractError as e:
        return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
    return _errors_to_issues("bo-guard-spec-rewrite-graph", graph_result.get("errors", []))


def _content_issues(ctx: WriteContext) -> list[ValidationIssue]:
    normalized = ctx.path.replace("\\", "/")
    if normalized.startswith(SCHEDULES_PREFIX):
        return _schedule_rewrite_issues(ctx)
    if normalized.startswith(SPECS_PREFIX):
        return _spec_rewrite_issues(ctx)
    return []


def _path_mutation_issues(ctx: PathMutationContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for schedule_path in _affected_paths(ctx.path, SCHEDULES_PREFIX):
        try:
            result = bo_contract.preflight_schedule_move(schedule_path)
        except bo_contract.BOContractError as e:
            issues.append(ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e)))
            continue
        issues.extend(_errors_to_issues("bo-guard-schedule-move", result.get("errors", [])))

    for spec_path in _affected_paths(ctx.path, SPECS_PREFIX):
        build_id = _build_id_from_spec_path(spec_path)
        try:
            result = bo_contract.preflight_ids([build_id])
        except bo_contract.BOContractError as e:
            issues.append(ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e)))
            continue
        status = (result.get("results") or {}).get(build_id)
        if status is not None and status not in _freely_mutable_statuses():
            issues.append(ValidationIssue(
                "bo-guard-spec-move", "reject",
                f"{build_id!r} has status {status!r} — {ctx.operation}ing its spec risks orphaning a "
                "bound or closed build (schedule_unbound-style regression)",
            ))
    return issues


def _inbound_move_issues(ctx: PathMutationContext) -> list[ValidationIssue]:
    """Validate whatever bytes/artifacts WOULD land at the DESTINATION,
    independent of whether the source is also a BO path (B8,
    codex-review-bo-authoring-contract-v2: "a BO-path source does not exempt
    the BO-path destination" -- e.g. a specs/ -> schedules/ or schedules/ ->
    specs/ move must validate the destination type against the bytes that
    will land there, not just the source's own status). Also covers the
    original B1 case of an inbound move from a non-BO path. Validated as if
    it were a fresh write at the destination path -- move never changes
    content, so the source's current on-disk bytes are exactly what would
    land there.
    """
    destination = ctx.destination
    if not destination:
        return []
    content = _read_vault_text(ctx.path)
    if content is not None:
        synthetic_ctx = WriteContext(path=destination, old_content=None, new_content=content, tool=ctx.operation)
        return _content_issues(synthetic_ctx)

    # Directory move/rename: `_read_vault_text` returns None both for an
    # unreadable/missing path AND for a directory -- enumerate every file
    # that would land under `destination` and validate each as a fresh
    # write at its mapped path (B8's directory-move gap). Fails closed
    # (reject) for any nested artifact that cannot itself be evaluated,
    # rather than silently skipping it.
    from . import config

    source_full = (config.VAULT_PATH / ctx.path).resolve()
    if not source_full.is_dir():
        return []  # genuinely missing/unreadable source -- nothing to validate

    issues: list[ValidationIssue] = []
    source_prefix = ctx.path.replace("\\", "/").rstrip("/")
    dest_prefix = destination.replace("\\", "/").rstrip("/")
    for rel_file in _files_under(ctx.path):
        mapped = dest_prefix + rel_file[len(source_prefix):]
        nested_content = _read_vault_text(rel_file)
        if nested_content is None:
            issues.append(ValidationIssue(
                "bo-guard-inbound-directory-move", "reject",
                f"cannot evaluate {rel_file!r}, which would land at {mapped!r} under "
                f"{destination!r} -- failing closed rather than silently skipping it",
            ))
            continue
        nested_ctx = WriteContext(path=mapped, old_content=None, new_content=nested_content, tool=ctx.operation)
        issues.extend(_content_issues(nested_ctx))
    return issues


def evaluate_content(ctx: WriteContext) -> BOGateResult:
    mode = get_mode()
    if mode == "off" or not _is_bo_path(ctx.path):
        return BOGateResult(mode=mode, issues=[])
    issues = _content_issues(ctx)
    _log_issues(mode, ctx.path, ctx.tool, issues)
    return BOGateResult(mode=mode, issues=issues)


def evaluate_path_mutation(ctx: PathMutationContext) -> BOGateResult:
    mode = get_mode()
    if mode == "off":
        return BOGateResult(mode=mode, issues=[])

    source_is_bo = _is_bo_path(ctx.path)
    dest_is_bo = bool(ctx.destination) and _is_bo_path(ctx.destination)
    if not source_is_bo and not dest_is_bo:
        return BOGateResult(mode=mode, issues=[])

    issues: list[ValidationIssue] = []
    if source_is_bo:
        issues.extend(_path_mutation_issues(ctx))
    if dest_is_bo:
        # Destination authority is evaluated independently of whether the
        # source is also a BO path -- a BO-path source is not an exemption
        # for the BO-path destination (B8).
        issues.extend(_inbound_move_issues(ctx))

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
