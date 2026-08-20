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

opus-review-bo-authoring-contract-v4 (2026-08-18) found a BLOCKER and two
HIGHs against the REAL adapter (the repo's own tests monkeypatch
bo_contract wholesale, which hid these), plus two lower-severity findings,
all closed in vault-bo-authoring-guard-remediation-v5, still shadow-only:

  - BL-1 (BLOCKER): vault.py used to construct WriteContext/
    PathMutationContext from the caller's RAW path string, while the actual
    filesystem target was resolve_vault_path(...)'s NORMALISED result --
    aliases like `//`, `./`, `/./`, and a vault-internal symlinked
    directory collapsed to the canonical path at the filesystem layer but
    not at `_is_bo_path`'s literal-prefix check, so they skipped guard
    evaluation entirely (zero adapter invocations) through vault_write,
    vault_move, vault_delete and every tool that funnels through them.
    vault.py now derives the WriteContext/PathMutationContext path (and
    move destination) from resolve_vault_path's own resolved, symlink-
    collapsed result via `canonical_vault_relative()`, not the caller's
    string -- one normalisation point closes every alias witness.
  - HI-1: no guard evaluation asserted the adapter's schema/contract
    version before trusting its preflight/graph results -- only
    _freely_mutable_statuses' internal fallback touched check_version, and
    it silently substituted a hardcoded status set on ANY adapter failure
    (including version drift), converting an untrustworthy adapter into
    permission. evaluate_content/evaluate_path_mutation now call
    check_version() fresh on every evaluation via _adapter_version_issues(),
    short-circuiting to a reject issue on drift/unavailability; the
    fallback allowlist is gone entirely -- an adapter failure anywhere in
    this module is now always a reject issue, never a silent guess.
  - HI-2: _spec_rewrite_schedule_graph_issues returned `[]` (no issue) for
    an un-ingested build (no DB row yet), a DB-bound schedule missing on
    disk, an unparseable bound schedule, or a bound schedule whose builds:
    list doesn't actually contain the build's entry -- each is now a
    reject-severity finding instead of a silent skip
    (_revalidate_schedule_with_proposed_spec). Referring schedules are also
    now discovered by scanning disk (_disk_schedules_referencing), not just
    the DB, so a spec/schedule pair authored just ahead of the
    orchestrator's next ingest scan is still covered.
  - MD-1: an undecodable/binary file moved into a BO destination returned
    `[]` from _inbound_move_issues (mis-handled as "directory branch found
    nothing, therefore nothing to validate") instead of failing closed like
    every other unevaluable-artifact case in this module. Now rejects.
  - LO-1: _affected_paths looked up a SUBDIRECTORY of specs/ or schedules/
    as one exact path (matching nothing) instead of enumerating the files
    nested inside it, unlike its handling of the top-level directory
    itself or an ancestor of it. Now recurses into a subdirectory the same
    way.

  Deliberately NOT changed in this build: BO_PATH_GUARD_MODE stays shadow
  on all three live services; MD-2 (build_generator.generate_build()'s own
  authoring path, which never went through this guard) is explicitly out
  of scope, owned by bo-authoring-contract-runtime-remediation-v5 instead;
  LO-2 (authoring_contract.py's own exception handling) is likewise BO-repo
  scope, not this module's.

  Race-window note (required review item 9): bo_guard reads a referring
  schedule's bytes via _read_vault_text without holding a lock on that
  file (it is a different resolved path from the one vault.py's caller
  holds a lock on). vault.write_file_atomic now re-runs bo_guard.enforce()
  a second time immediately before its filesystem commit, which re-reads
  the referring schedule's then-current bytes and narrows this window
  considerably. This is NOT a provable elimination of the race -- a writer
  landing in the few remaining lines between that second check and the
  commit is not caught. No stronger guarantee is claimed; the opus-v4
  review itself could not reproduce a committing bypass of this exact
  mechanism either (it required monkeypatching internals to force one).

opus-review-bo-authoring-contract-v5 (2026-08-18) found one further BLOCKER
and three further HIGHs against the real adapter and the real deployed
schedule corpus (v5's remediation had only been tested against synthetic
fixtures), plus three MEDIUM/LOW findings, all closed in
vault-bo-authoring-guard-remediation-v6, still shadow-only:

  - BL5-1 (BLOCKER): this module's own parse_schedule_builds -- a naive
    whole-body yaml.safe_load -- could not parse 67 of 120 real deployed
    schedule files (56%) because the real corpus's `>` blockquote
    program-note line parses as a YAML block-scalar indicator under a
    whole-body parse. That single defect fails open for un-ingested builds
    (a graph-breaking spec rewrite could commit) and over-blocks benign
    edits to DB-bound specs referencing a real-shaped schedule. Removed
    entirely; schedule content is now always parsed via
    bo_contract.parse_schedule_document, a thin reference (not a vendored
    copy -- see bo_contract.py's _schedule_parser) to the BO repo's own
    contract.parse_schedule_document, the same canonical parser
    schedule_loader.py/db.py/build_generator.py/project_resolve.py already
    use. There is now exactly one schedule parser in this whole system.
  - HI5-1: move_path/delete_path never got the activation-boundary
    re-enforce() call write_file_atomic already had -- a directory-move's
    validation enumerates and reads every nested file, but nothing
    re-checked those files' bytes immediately before the actual
    shutil.move(), so a concurrent write landing mid-enumeration could land
    unvalidated, malformed bytes under BO authority. Reproduced 12/12 with
    real cooperating processes. Both functions now re-run
    bo_guard.enforce_path_mutation() immediately before their filesystem
    mutation, matching write_file_atomic's own pattern -- same "narrows,
    does not provably eliminate" caveat as the race-window note above.
  - HI5-3: _revalidate_schedule_with_proposed_spec's whole-graph check
    always resolved live DB state for every sibling node in the referring
    schedule (there is no JSON-safe way to pass BO's own
    existing_state_lookup=lambda _bid: None over the adapter's CLI
    boundary), so ANY terminal/dispatched/duplicate-authority sibling
    sitting unchanged in the same schedule file could block an otherwise
    benign edit -- contradicting authoring_contract's own documented
    semantics for exactly this graph-context use (its
    whole_graph_errors_for_new_build deliberately disables that lookup).
    Fixed by filtering the adapter's returned errors client-side: a
    terminal_id_reuse/duplicate_authority/frozen_contract_mutation finding
    attributed to a DIFFERENT build_id than the one actually being
    rewritten is dropped; the same finding attributed to build_id itself
    still applies in full. (vault-bo-authoring-cache-remediation-v7 later
    generalized this filter -- see _revalidate_schedule_with_proposed_spec's
    own docstring/comment -- to drop ANY error attributed to a different
    build_id, closing the residual sibling-content-shape bleed the v6 build
    flagged but did not fix.)
  - MD5-1: _path_mutation_issues lacked the disk-based referring-schedule
    discovery the rewrite path already had (HI-2b) -- an un-ingested
    (no DB row) spec already declared in an on-disk schedule could be moved
    or deleted with no check at all. Now also calls
    _disk_schedules_referencing for every affected spec path, symmetric
    with the rewrite path.
  - LO5-1: _read_vault_text had no vault-root containment check (unlike
    vault.resolve_vault_path), so a crafted spec_path in a schedule entry
    could read an arbitrary host file's bytes into the adapter's evaluation
    payload. Now refuses (returns None) anything that resolves outside the
    vault root.
  - LO5-2: write_file_atomic's activation-boundary re-enforce() reused the
    ORIGINAL write_ctx, whose old_content was captured at the first read --
    defeating the point of re-checking this path's own old_content a second
    time. Now rebuilt from the bytes re-read immediately before this second
    check.
  - LO5-3: v4-era untracked scratch probe scripts (item11.py, item67.py,
    mover.py) removed from the live deployed checkout -- confirmed unused
    (no references anywhere in src/, tests/, scripts/, or pyproject.toml)
    before deletion.

  Deliberately NOT changed in this build: BO_PATH_GUARD_MODE stays shadow
  on all three live services; no bulk rewrite of historical schedule/spec
  authority was performed to make MD5-2's real-corpus report look better
  (see that report in this build's output log); no Build Orchestrator repo
  file (authoring_contract.py/schedule_loader.py/build_generator.py) was
  modified -- BL5-1's fix references contract.py by path, which already had
  zero repo-internal imports by design, precisely so an external consumer
  like this one could load it without duplicating its logic or waiting on a
  BO-repo change.

vault-bo-authoring-enforce-v2 (2026-08-19) verified this module's guard
implementation itself is sound and reviewed, then re-ran
vault-bo-authoring-cache-remediation-v7's real-corpus readiness report and
found every then-current pre-dispatch build was blocked by its own genuine
missing_summary_instruction defect, and 10/13 also carried an unrelated,
never-sole-blocking mixed_project_schedule finding from the legacy
2026-W34-bs-brain-hardening-v1.yaml schedule (which mixes the
mcp-infrastructure and obsidian-web-mcp projects) -- a real readiness gap,
not a guard-correctness defect, so activation was deliberately stopped
rather than freezing normal editing of every real pre-dispatch build.

vault-bo-authoring-enforcement-readiness-v1 (2026-08-21) closes that
readiness gap with a narrow, mechanically-derived compatibility rule in
_revalidate_schedule_with_proposed_spec (the spec-rewrite graph-revalidation
path only -- _schedule_rewrite_issues, which handles genuine schedule
mutations, is untouched and remains fully strict): before substituting the
proposed spec bytes, the function now also computes a BASELINE graph result
(mode="compat_existing", no substitution -- exactly what's on disk today)
and, when converting the PROPOSED graph result's errors to issues, drops any
mixed_project_schedule finding whose message is byte-identical to one
already present in that baseline. The message embeds the exact sorted() set
of resolved projects, so this is a pure "unchanged legacy condition, not
introduced or worsened by this edit" test -- any edit that changes the
resulting project set (a genuinely new mix, a different project, one fewer
conflicting project) produces a different message and is never dropped.
Scoped to this single code only: dependency_cycle, unknown_project, and
every per-node content-shape code remain unconditionally blocking, matching
the existing HI5-3/cache-remediation-v7 sibling-attribution filter's own
precedent of narrow, mechanically-justified scope. Verified against the live
real corpus (not synthetic fixtures): the legacy schedule's mixed condition
is confirmed unchanged before/after for every currently-affected build, and
a real negative canary (mutating a real build's own resolved project) still
rejects, with a NEW mixed_project_schedule message reflecting the changed
project set. Still shadow-only; BO_PATH_GUARD_MODE activation is this same
build's own later phase, gated on independent review of this exact change
plus a hash-pinned summary-instruction backfill manifest for the remaining
real defect.
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

# The authoritative freely-mutable-status set is sourced from the adapter's
# own op=version vocabulary (known_statuses - terminal_statuses -
# dispatched_statuses), not hardcoded here (codex-review-bo-authoring-
# contract-v1, B5: "the status classification should be owned by and
# returned from the contract adapter"). There is deliberately no hardcoded
# fallback: opus-review-bo-authoring-contract-v4 (HI-1) found the previous
# fallback-on-adapter-failure behaviour could silently convert an adapter
# version/availability failure into permission for a mutation that should
# have been rejected. A failure here now propagates to the caller, which
# must turn it into a reject-severity issue, never substitute a guess.
_freely_mutable_statuses_cache: set | None = None


def _freely_mutable_statuses() -> set:
    """Cached (process-lifetime) lookup of the adapter's own
    freely-mutable-status vocabulary. A restart is required to pick up a
    change in the adapter's vocabulary, matching how EXPECTED_SCHEMA_VERSION/
    EXPECTED_CONTRACT_VERSION drift already requires a restart. Raises
    bo_contract.BOContractError on adapter failure -- callers must catch it
    and produce a reject issue (see _spec_rewrite_issues/_path_mutation_issues),
    never silently proceed with a guessed status set.
    """
    global _freely_mutable_statuses_cache
    if _freely_mutable_statuses_cache is not None:
        return _freely_mutable_statuses_cache
    _freely_mutable_statuses_cache = bo_contract.freely_mutable_statuses()
    return _freely_mutable_statuses_cache


def _adapter_version_issues() -> list[ValidationIssue]:
    """Assert the authoring-contract adapter's schema/contract version before
    trusting any preflight/graph result from it (HI-1,
    opus-review-bo-authoring-contract-v4, required review item 9). Unlike
    _freely_mutable_statuses()'s process-lifetime cache, this runs fresh on
    every guard evaluation -- version drift is a real, time-varying signal
    (an adapter can be redeployed while this server keeps running), not
    something safe to cache for a process's whole lifetime. A drifted or
    unavailable adapter is a reject-severity finding, never a silently
    substituted fallback.
    """
    try:
        bo_contract.check_version()
    except bo_contract.BOContractError as e:
        return [ValidationIssue("bo-guard-adapter-version", "reject", str(e))]
    return []


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


def schedule_builds_from_content(content: str, source_name: str = "<schedule>") -> list[dict] | None:
    """Extract the `builds:` list from a schedule file's body via
    `bo_contract.parse_schedule_document` -- the one canonical schedule
    parser (BL5-1, opus-review-bo-authoring-contract-v5). This module
    previously re-implemented this parse locally
    (``bo_guard.parse_schedule_builds``, a naive whole-body
    ``yaml.safe_load``); that second implementation could not parse 67 of
    120 real deployed schedule files (56%) because the real corpus's `>`
    blockquote program-note line parses as a YAML block-scalar indicator
    under a whole-body parse, and disagreed with the canonical parser on the
    result -- exactly the "two schedule parsers that can drift" failure mode
    this fix removes.

    Returns None if the content is malformed (a per-content ValueError from
    the canonical parser) or has no `builds:` list -- callers treat that as
    "cannot evaluate", not as "zero builds". Raises BOContractError if the
    canonical parser module itself is unavailable -- that is a systemic
    adapter failure, not a per-content one, and callers must fail closed on
    it rather than silently treating it the same as "no builds".
    """
    try:
        data = bo_contract.parse_schedule_document(content, source_name=source_name)
    except ValueError:
        return None
    builds = data.get("builds") if isinstance(data, dict) else None
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

    Vault-root containment (LO5-1, opus-review-bo-authoring-contract-v5):
    unlike vault.resolve_vault_path, this function previously had no check
    that the resolved path stays inside the vault root -- a crafted
    spec_path in a schedule entry (e.g. "../../../../etc/hostname") would
    read an arbitrary host file's bytes into the adapter's evaluation
    payload. Refuses (returns None, same as any other unreadable path) for
    anything that resolves outside the vault root.
    """
    from . import config

    try:
        full = (config.VAULT_PATH / relative_path).resolve()
        vault_root = config.VAULT_PATH.resolve()
        if full != vault_root and not str(full).startswith(str(vault_root) + os.sep):
            logger.warning("bo-guard refused to read out-of-vault path %r (resolved %s)", relative_path, full)
            return None
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

    Three cases: `path` is a file under `prefix` (returns that one path);
    `path` is itself a SUBDIRECTORY of `prefix` (returns every file
    currently under it on disk, recursively -- LO-1,
    opus-review-bo-authoring-contract-v4: a subdirectory-shaped path used to
    be looked up as one exact authority path, matching nothing, rather than
    enumerating the schedule/spec files nested inside it); `path` IS
    `prefix`'s directory itself, or an ancestor of it (returns every file
    currently under `prefix` on disk -- the bulk case); otherwise empty.
    """
    normalized = path.replace("\\", "/").rstrip("/")
    prefix_no_slash = prefix.rstrip("/")
    if (prefix_no_slash + "/").startswith(normalized + "/"):
        return _files_under(prefix)
    if normalized.startswith(prefix):
        from . import config

        full = (config.VAULT_PATH / normalized).resolve()
        if full.is_dir():
            return _files_under(normalized)
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

    try:
        new_builds = schedule_builds_from_content(ctx.new_content, source_name=ctx.path)
    except bo_contract.BOContractError as e:
        return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
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
    # The parser module is already proven loadable by the new_builds call
    # above (cached for the process lifetime) -- only a malformed old_content
    # can make this return None, which is the pre-existing "no prior builds
    # to diff against" degradation.
    schedule_project = _schedule_project_from_content(ctx.new_content)
    old_builds = schedule_builds_from_content(ctx.old_content, source_name=ctx.path) if ctx.old_content else None
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
    if status is not None:
        try:
            freely_mutable = _freely_mutable_statuses()
        except bo_contract.BOContractError as e:
            return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
        if status not in freely_mutable:
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


def _disk_schedules_referencing(build_id: str) -> list[str]:
    """Vault-relative paths of every on-disk schedule file under
    SCHEDULES_PREFIX whose builds: list contains an entry with this id,
    discovered by reading and parsing the files directly rather than the BO
    database.

    Covers the case where a build has not yet been ingested into the BO
    database (so bo_contract.preflight_source_schedule finds nothing via its
    DB lookup) but is already declared in a schedule file on disk (HI-2b,
    opus-review-bo-authoring-contract-v4). A schedule file that cannot itself
    be read/parsed is skipped here, not failed closed -- this is a discovery
    pass over files not yet known to reference build_id at all, distinct
    from _revalidate_schedule_with_proposed_spec's fail-closed handling of a
    schedule ALREADY known (from the DB or this scan) to reference it.
    """
    matches: list[str] = []
    for schedule_path in _files_under(SCHEDULES_PREFIX):
        content = _read_vault_text(schedule_path)
        if content is None:
            continue
        # A malformed individual schedule is skipped here (discovery pass,
        # per this function's own contract above); an unavailable parser
        # module (BOContractError) is systemic and propagates to the caller,
        # which fails closed rather than silently treating it as "no match".
        builds = schedule_builds_from_content(content, source_name=schedule_path)
        if not builds:
            continue
        for entry in builds:
            if isinstance(entry, dict) and entry.get("id") == build_id:
                matches.append(schedule_path)
                break
    return matches


def _revalidate_schedule_with_proposed_spec(
    schedule_path: str, build_id: str, new_spec_content: str
) -> list[ValidationIssue]:
    """Revalidate one schedule's complete graph with build_id's node
    substituted for the PROPOSED spec bytes, failing closed (reject) at
    every step that opus-review-bo-authoring-contract-v4's HI-2 found
    returning `[]` (silently skipping revalidation) instead: an unreadable/
    missing schedule file, an unparseable schedule body, or a schedule that
    (despite being a known binding) does not actually list build_id in its
    builds:.
    """
    schedule_content = _read_vault_text(schedule_path)
    if schedule_content is None:
        return [ValidationIssue(
            "bo-guard-spec-rewrite-graph", "reject",
            f"{build_id!r} is referenced by schedule {schedule_path!r}, which is missing or "
            "unreadable -- failing closed rather than skipping graph revalidation",
        )]

    try:
        builds = schedule_builds_from_content(schedule_content, source_name=schedule_path)
    except bo_contract.BOContractError as e:
        return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
    if builds is None:
        return [ValidationIssue(
            "bo-guard-spec-rewrite-graph", "reject",
            f"{build_id!r} is referenced by schedule {schedule_path!r}, which does not parse as a "
            "valid builds: list -- failing closed rather than skipping graph revalidation",
        )]

    schedule_project = _schedule_project_from_content(schedule_content)
    nodes = _nodes_for_schedule_builds(builds, schedule_path, schedule_project)

    if not any(node["build_id"] == build_id for node in nodes):
        return [ValidationIssue(
            "bo-guard-spec-rewrite-graph", "reject",
            f"{build_id!r} is referenced by schedule {schedule_path!r} but no entry with that id is "
            "present in its builds: list -- failing closed rather than validating the graph without "
            "the proposed bytes",
        )]

    # Baseline (vault-bo-authoring-enforcement-readiness-v1, Phase 1): the
    # graph exactly as it exists on disk today, build_id's spec bytes
    # untouched -- read BEFORE the substitution below so it reflects the
    # unedited corpus. Used only to detect whether a mixed_project_schedule
    # finding already existed independent of this edit. mode="compat_existing"
    # is the adapter's own "read-only auditing of the historical corpus" mode
    # (authoring_contract.py's own module docstring) -- exactly the semantics
    # needed for a baseline snapshot; new_ids is intentionally omitted so
    # every mixed_project_schedule instance lands in `warnings`, not `errors`
    # (this call is never used to block anything itself).
    try:
        baseline_result = bo_contract.validate_graph(nodes, mode="compat_existing")
    except bo_contract.BOContractError as e:
        return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
    baseline_mixed_messages = {
        e.get("message")
        for e in (baseline_result.get("errors", []) + baseline_result.get("warnings", []))
        if e.get("code") == "mixed_project_schedule"
    }

    # Substitute the proposed new spec content for this build_id's node so
    # the graph is validated against the bytes that WOULD exist after this
    # write commits, not what's currently on disk under specs/.
    for node in nodes:
        if node["build_id"] == build_id:
            node["spec_markdown"] = new_spec_content

    try:
        graph_result = bo_contract.validate_graph(nodes, mode="strict_new", new_ids=[build_id])
    except bo_contract.BOContractError as e:
        return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
    # HI5-3 (opus-review-bo-authoring-contract-v5): the adapter's
    # validate_graph always resolves live DB state for every node (there is
    # no JSON-safe way to pass BO's own existing_state_lookup=lambda _bid:
    # None over the CLI boundary the way authoring_contract's own
    # whole_graph_errors_for_new_build does for exactly this call shape). A
    # sibling elsewhere in the same schedule that happens to be
    # terminal/dispatched, or already bound to a different schedule, is pure
    # GRAPH CONTEXT for revalidating build_id's own proposed bytes -- it must
    # never block this write merely for being present as context, matching
    # BO's own documented semantics. HI5-3 originally scoped this filter to
    # just the three DB-state codes ({"terminal_id_reuse", "duplicate_
    # authority", "frozen_contract_mutation"}).
    #
    # vault-bo-authoring-cache-remediation-v7 (residual finding flagged by
    # vault-bo-authoring-guard-remediation-v6's real-corpus report): that
    # narrow scoping left every OTHER per-node error code -- in particular
    # missing_summary_instruction/summary_token_mismatch/unknown_tier/etc,
    # produced by preflight_spec_validate's own content-shape checks for
    # EVERY node in the schedule, not just build_id's -- unfiltered when
    # attributed to a sibling, so a sibling's own unrelated authoring defect
    # bled into build_id's own edit as a rejecting issue. Confirmed still
    # reproducing against the real corpus (2026-08-20): a benign edit to any
    # of several real pre-dispatch specs in a large shared schedule surfaced
    # a dozen-plus sibling-attributed `missing_summary_instruction` findings
    # that have nothing to do with the edit being made.
    #
    # Generalized the filter the same way authoring_contract._sound_sibling_
    # nodes/whole_graph_errors_for_new_build already treat this exact
    # "one build's own promotion/edit surrounded by sibling graph context"
    # shape: drop ANY error attributed to a build_id other than the one
    # actually being rewritten, regardless of code. This is safe specifically
    # because this function only ever substitutes build_id's proposed SPEC
    # CONTENT into an otherwise-unchanged schedule graph (it never adds,
    # removes, or reorders schedule entries) -- so every error attributed to
    # a different build_id necessarily describes a pre-existing condition of
    # that sibling's own entry, not something build_id's edit introduced or
    # worsened. Errors with no per-node build_id attribution at all --
    # mixed_project_schedule (attributed via `path=schedule_path`) and
    # dependency_cycle (attributed to neither) -- describe the WHOLE
    # resulting graph and are never filtered by this rule, so a genuinely
    # invalid graph (e.g. mixed-project) still rejects regardless of which
    # node the adapter happened to name.
    filtered_errors = [
        e for e in graph_result.get("errors", [])
        if e.get("build_id") is None or e.get("build_id") == build_id
    ]

    # Phase 1 (vault-bo-authoring-enforcement-readiness-v1): a
    # mixed_project_schedule finding whose message is BYTE-IDENTICAL to one
    # already present in the baseline (unedited) graph describes a
    # pre-existing legacy condition this content-only edit does not touch --
    # it must not, by itself, make an otherwise-valid edit impossible.
    # Scoped narrowly to this one code (dependency_cycle/unknown_project/
    # every other graph-wide or per-node code is never touched by this
    # filter and remains unconditionally blocking) and derived purely by
    # diffing the adapter's own baseline-vs-proposed output -- never by
    # name-matching a specific schedule file or hardcoding a project list.
    # The message embeds the schedule path and the exact sorted() set of
    # resolved projects, so ANY change this edit makes to that set (a new
    # mix, a different project, one fewer conflicting project) produces a
    # different message and is NOT filtered -- it stays blocking.
    filtered_errors = [
        e for e in filtered_errors
        if not (e.get("code") == "mixed_project_schedule" and e.get("message") in baseline_mixed_messages)
    ]
    return _errors_to_issues("bo-guard-spec-rewrite-graph", filtered_errors)


def _spec_rewrite_schedule_graph_issues(build_id: str, ctx: WriteContext) -> list[ValidationIssue]:
    try:
        sched_result = bo_contract.preflight_source_schedule([build_id])
    except bo_contract.BOContractError as e:
        return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
    db_bound_schedule = (sched_result.get("results") or {}).get(build_id)

    # Combine the DB's own binding (if any) with every on-disk schedule that
    # references this build id (HI-2b) -- an un-ingested build can be
    # declared in a schedule file before the orchestrator's next scan picks
    # it up, and that window must not be exempt from revalidation.
    try:
        disk_referenced = _disk_schedules_referencing(build_id)
    except bo_contract.BOContractError as e:
        return [ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e))]
    candidate_schedules = sorted({
        *( [db_bound_schedule] if db_bound_schedule else [] ),
        *disk_referenced,
    })
    if not candidate_schedules:
        return []  # genuinely not referenced anywhere -- nothing to revalidate

    issues: list[ValidationIssue] = []
    for schedule_path in candidate_schedules:
        issues.extend(_revalidate_schedule_with_proposed_spec(schedule_path, build_id, ctx.new_content))
    return issues


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
        if status is not None:
            try:
                freely_mutable = _freely_mutable_statuses()
            except bo_contract.BOContractError as e:
                issues.append(ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e)))
                continue
            if status not in freely_mutable:
                issues.append(ValidationIssue(
                    "bo-guard-spec-move", "reject",
                    f"{build_id!r} has status {status!r} — {ctx.operation}ing its spec risks orphaning a "
                    "bound or closed build (schedule_unbound-style regression)",
                ))
                continue
        # MD5-1 (opus-review-bo-authoring-contract-v5): the rewrite path
        # already discovers referring schedules by scanning disk, not just
        # the DB (_disk_schedules_referencing, HI-2b) -- the move/delete path
        # did not, so a build with no DB row yet (status is None above, or
        # simply never checked because it IS freely-mutable) could be moved
        # or deleted out from under an on-disk schedule that already
        # declares it, orphaning that reference before the orchestrator's
        # next ingest scan ever saw it.
        try:
            referenced_by = _disk_schedules_referencing(build_id)
        except bo_contract.BOContractError as e:
            issues.append(ValidationIssue("bo-guard-adapter-unavailable", "reject", str(e)))
            continue
        if referenced_by:
            issues.append(ValidationIssue(
                "bo-guard-spec-move", "reject",
                f"{build_id!r} is referenced by on-disk schedule(s) {referenced_by!r} — "
                f"{ctx.operation}ing its spec would orphan that reference",
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
        if source_full.is_file():
            # A binary/undecodable file moving into a BO destination (MD-1,
            # opus-review-bo-authoring-contract-v4): _read_vault_text
            # returned None because the bytes exist but aren't valid UTF-8
            # text, not because the source is missing. The module's own
            # policy is to fail closed for anything it cannot evaluate --
            # this case was falling through that policy instead of hitting it.
            return [ValidationIssue(
                "bo-guard-inbound-move-undecodable", "reject",
                f"{ctx.path!r} cannot be decoded as UTF-8 text, so its content cannot be validated "
                f"before it lands at {destination!r} under BO authority -- failing closed",
            )]
        return []  # genuinely missing source -- nothing to validate

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
    # Assert the adapter's schema/contract version before trusting any of its
    # preflight/graph results for this evaluation (HI-1). A drifted or
    # unavailable adapter short-circuits straight to a reject issue rather
    # than proceeding to call it further.
    issues = _adapter_version_issues()
    if not issues:
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

    issues = _adapter_version_issues()
    if not issues:
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
