"""Build Orchestrator authoring tools (vault-bo-authoring-mcp-v1).

bo_validate_build_graph / bo_create_build / bo_create_chain. All three delegate
BO schema validation and rendering entirely to the authoring contract adapter
(bo_contract.py -> authoring_contract.py's JSON CLI) -- this module only:

  - shapes typed tool inputs into the adapter's node/spec JSON shape,
  - enforces the write ordering invariant (validate the whole graph -> write
    every new spec -> write/replace the schedule last, as the activation
    boundary), so a pre-schedule failure leaves any already-written spec
    orphaned (inert -- not referenced by any schedule entry) rather than
    producing a half-activated graph,
  - fails closed if the adapter is unavailable or a schema-version mismatch is
    detected (bo_contract.BOContractError) -- never guesses at validity.

v1 is create-only, matching the plan's scope (no bo_update_build): builds are
appended to an EXISTING schedule file, never created fresh -- the exact,
already-proven append operation build_generator.generate_build() uses
(a fully-rendered, structured YAML entry block appended after the existing
content, then the whole result re-parsed and asserted correct before anything
is written -- never hand-indented YAML).
"""

from __future__ import annotations

import json
import logging

import frontmatter

from .. import bo_contract
from ..bo_guard import SPECS_PREFIX, parse_schedule_builds
from ..vault import RevisionConflictError, conflict_payload, read_file, write_file_atomic

logger = logging.getLogger(__name__)

REQUIRED_BUILD_FIELDS = ("build_id", "title", "body_markdown", "tier", "project")


class BOToolError(Exception):
    """Internal signal for a clean, structured tool-level failure -- never a
    validation failure (those come back as ok:false with errors/warnings)."""


def _normalize_build(build: dict) -> dict:
    if not isinstance(build, dict):
        raise BOToolError(f"each build must be an object, got {type(build).__name__}")
    for required in REQUIRED_BUILD_FIELDS:
        if not build.get(required):
            raise BOToolError(f"build is missing required field {required!r}: {build!r}")
    return build


def _spec_path_for(build_id: str) -> str:
    return f"{SPECS_PREFIX}{build_id}.md"


def _schedule_entry_for(build: dict) -> dict:
    entry = {
        "id": build["build_id"],
        "title": build["title"],
        "description": build.get("description") or build["title"],
        "run_when": build.get("run_when") or "no deps — dispatch immediately",
        "tier": build["tier"],
        "depends_on": build.get("depends_on") or [],
        "spec_path": _spec_path_for(build["build_id"]),
        "project": build["project"],
    }
    for optional in ("risk_domain", "blast_radius", "reversible", "shadowable", "engine", "notes"):
        if build.get(optional) is not None:
            entry[optional] = build[optional]
    return entry


def _render_spec_item(build: dict, schedule_entry: dict) -> dict:
    item = {
        "build_id": build["build_id"],
        "title": build["title"],
        "body_markdown": build["body_markdown"],
        "tier": build["tier"],
        "project": build["project"],
        "status": build.get("status") or "ready",
        "schedule_entry": schedule_entry,
    }
    for optional in ("risk_domain", "blast_radius", "reversible", "shadowable", "program",
                      "tags", "created", "completion_contract"):
        if build.get(optional) is not None:
            item[optional] = build[optional]
    return item


def _existing_schedule_nodes(schedule_path: str, new_build_ids: set) -> tuple[list[dict], str | None]:
    """Load every entry already in schedule_path's builds: list (excluding
    ids the caller is about to (re)supply) as additional graph nodes, each
    paired with its own on-disk spec content.

    Without this, whole-graph checks (mixed-project, duplicate-id) only ever
    saw the newly-proposed nodes -- a strict-new build appended to a schedule
    that already contained a different project's build validated cleanly
    because the existing entry was never part of the graph being checked
    (codex-review-bo-authoring-contract-v1, B2). Returns (nodes,
    schedule_project) -- schedule_project is None for a brand-new schedule.
    """
    try:
        content, _ = read_file(schedule_path)
    except FileNotFoundError:
        return [], None

    project = None
    try:
        project = frontmatter.loads(content).metadata.get("project")
    except Exception:
        pass

    nodes = []
    for entry in parse_schedule_builds(content) or []:
        if not isinstance(entry, dict) or entry.get("id") in new_build_ids:
            continue
        spec_path = entry.get("spec_path")
        spec_markdown = ""
        if isinstance(spec_path, str):
            try:
                spec_markdown, _ = read_file(spec_path)
            except FileNotFoundError:
                spec_markdown = ""
        nodes.append({
            "build_id": entry.get("id"),
            "schedule_entry": entry,
            "spec_markdown": spec_markdown,
            "schedule_path": schedule_path,
            "schedule_project": project,
        })
    return nodes, project


def _prepare_graph(builds: list[dict], schedule_path: str, mode: str) -> dict:
    """Shape + render + validate a proposed graph. Never writes anything.

    Validates the COMPLETE resulting graph -- every entry already in
    schedule_path plus the newly-proposed ones -- not just the newly-proposed
    rows in isolation (B2). `mode` (and its strict-new-only checks) applies
    only to the newly-proposed build ids; pre-existing entries are still
    included in every cross-node check (duplicate id, mixed project,
    dependency/cycle) but evaluated leniently at the per-node level, via
    `bo_contract.validate_graph`'s `new_ids` parameter.

    Returns {"ok", "errors", "warnings", "nodes", "rendered", "version_info"}
    where "nodes"/"rendered" are only meaningful when "ok" is True, and
    contain ONLY the newly-proposed nodes (existing nodes are read-only graph
    context, never re-rendered or re-written). Raises BOToolError for a
    malformed tool input, bo_contract.BOContractError if the adapter itself
    is unavailable/wrong-version/failing (fail closed).
    """
    normalized_builds = [_normalize_build(dict(b)) for b in builds]
    # Duplicate build_id within this request is deliberately left to the adapter's
    # own validate_build_graph (duplicate_id_in_graph) rather than pre-checked
    # here -- one less place this repo would have to independently get right.

    version_info = bo_contract.check_version()

    new_build_ids = {b["build_id"] for b in normalized_builds}
    existing_nodes, schedule_project = _existing_schedule_nodes(schedule_path, new_build_ids)

    schedule_entries = {b["build_id"]: _schedule_entry_for(b) for b in normalized_builds}
    render_specs = [_render_spec_item(b, schedule_entries[b["build_id"]]) for b in normalized_builds]

    rendered = bo_contract.render_graph(render_specs)["rendered"]

    new_nodes = [
        {
            "build_id": b["build_id"],
            "schedule_entry": schedule_entries[b["build_id"]],
            "spec_markdown": rendered[b["build_id"]]["spec"],
            "schedule_path": schedule_path,
            "schedule_project": schedule_project,
        }
        for b in normalized_builds
    ]

    result = bo_contract.validate_graph(existing_nodes + new_nodes, mode=mode, new_ids=sorted(new_build_ids))
    return {
        "ok": bool(result.get("ok", False)),
        "errors": result.get("errors", []),
        "warnings": result.get("warnings", []),
        "nodes": new_nodes,
        "rendered": rendered,
        "version_info": version_info,
    }


def bo_validate_build_graph(builds: list[dict], schedule_path: str, mode: str = "strict_new") -> str:
    """Read-only preflight: validate a proposed build graph. Never writes anything."""
    try:
        prep = _prepare_graph(builds, schedule_path, mode)
    except BOToolError as e:
        return json.dumps({"ok": False, "error": str(e)})
    except bo_contract.BOContractError as e:
        return json.dumps({"ok": False, "error": str(e), "code": e.code})

    canonical_graph = [
        {
            "build_id": n["build_id"],
            "spec_path": n["schedule_entry"]["spec_path"],
            "project": n["schedule_entry"].get("project"),
            "tier": n["schedule_entry"]["tier"],
            "depends_on": n["schedule_entry"].get("depends_on", []),
        }
        for n in prep["nodes"]
    ]
    return json.dumps({
        "ok": prep["ok"],
        "schema_version": prep["version_info"].get("schema_version"),
        "contract_version": prep["version_info"].get("contract_version"),
        "mode": mode,
        "schedule_path": schedule_path,
        "errors": prep["errors"],
        "warnings": prep["warnings"],
        "canonical_graph": canonical_graph,
    })


def _activate(builds: list[dict], schedule_path: str, tool_name: str) -> str:
    """Shared create path for bo_create_build (1 build) and bo_create_chain (N builds).

    Always validates in "strict_new" mode -- `compat_existing` is a read-only
    audit/compatibility mode for the historical corpus and must not be
    caller-selectable on a path that writes new artifacts (B3: "compatibility
    mode is exposed on mutation tools ... allows new malformed artifacts").
    """
    try:
        prep = _prepare_graph(builds, schedule_path, "strict_new")
    except BOToolError as e:
        return json.dumps({"ok": False, "error": str(e), "activated": False})
    except bo_contract.BOContractError as e:
        return json.dumps({"ok": False, "error": str(e), "code": e.code, "activated": False})

    if not prep["ok"]:
        return json.dumps({
            "ok": False, "errors": prep["errors"], "warnings": prep["warnings"], "activated": False,
        })

    # Refuse to silently overwrite an existing spec file -- even an orphaned
    # one never ingested as a task -- matching build_generator.generate_build()'s
    # established "refusing to overwrite existing spec" behaviour exactly.
    for node in prep["nodes"]:
        spec_path = node["schedule_entry"]["spec_path"]
        try:
            read_file(spec_path)
        except FileNotFoundError:
            continue
        return json.dumps({
            "ok": False,
            "error": f"refusing to overwrite existing spec at {spec_path!r}",
            "activated": False,
        })

    try:
        schedule_content, schedule_meta = read_file(schedule_path)
    except FileNotFoundError:
        return json.dumps({
            "ok": False,
            "error": (
                f"schedule_path {schedule_path!r} does not exist -- {tool_name} appends to an "
                "existing schedule only; it never creates a new schedule file"
            ),
            "activated": False,
        })

    new_schedule_content = schedule_content
    for node in prep["nodes"]:
        entry_text = prep["rendered"][node["build_id"]]["schedule_entry"]
        new_schedule_content = new_schedule_content.rstrip("\n") + "\n\n" + entry_text

    # Validate the fully-rendered result BEFORE writing anything -- a
    # half-written pair (spec written, schedule broken) is worse than writing
    # neither. Structural self-check only (parses, every new id present) --
    # not a second schema validator; BO semantics were already checked above.
    parsed_builds = parse_schedule_builds(new_schedule_content)
    if parsed_builds is None:
        return json.dumps({
            "ok": False,
            "error": "generated schedule content failed to re-parse before write -- refusing to write anything",
            "activated": False,
        })
    parsed_ids = {b.get("id") for b in parsed_builds if isinstance(b, dict)}
    for node in prep["nodes"]:
        if node["build_id"] not in parsed_ids:
            return json.dumps({
                "ok": False,
                "error": f"generated schedule content has no entry for {node['build_id']!r} after append",
                "activated": False,
            })

    created = []
    try:
        for node in prep["nodes"]:
            spec_path = node["schedule_entry"]["spec_path"]
            write_file_atomic(spec_path, node["spec_markdown"], create_dirs=True, tool=tool_name)
            created.append({
                "build_id": node["build_id"],
                "spec_path": spec_path,
                "project": node["schedule_entry"].get("project"),
                "depends_on": node["schedule_entry"].get("depends_on", []),
            })

        is_new, size = write_file_atomic(
            schedule_path, new_schedule_content, create_dirs=False, tool=tool_name,
            expected_revision=schedule_meta.get("revision"),
        )
    except (RevisionConflictError, ValueError, OSError) as e:
        # Every spec write before this point already landed -- they are orphaned
        # (inert: no schedule entry references them) rather than a half-activated
        # graph, since the schedule write is always the last, single operation.
        logger.error(f"{tool_name} activation failed after {len(created)} spec write(s): {e}")
        payload = {
            "ok": False,
            "error": str(e),
            "activated": False,
            "orphaned_specs": [c["spec_path"] for c in created],
        }
        if isinstance(e, RevisionConflictError):
            payload.update(conflict_payload(e))
        return json.dumps(payload)

    return json.dumps({
        "ok": True,
        "schema_version": prep["version_info"].get("schema_version"),
        "contract_version": prep["version_info"].get("contract_version"),
        "schedule_path": schedule_path,
        "created": created,
        "activation": {"schedule_path": schedule_path, "size": size, "created_new_file": is_new},
    })


def bo_create_build(build: dict, schedule_path: str) -> str:
    """Structured single-build create: validate, write the spec, then append the
    schedule entry as the activation boundary. Always strict_new -- see _activate."""
    return _activate([build], schedule_path, "bo_create_build")


def bo_create_chain(builds: list[dict], schedule_path: str) -> str:
    """Structured same-project multi-build chain create, including forward
    references (a later build's depends_on may name an earlier build in the
    same request). Validates the whole graph, writes every spec, then appends
    all schedule entries in one final schedule write. Always strict_new -- see _activate."""
    return _activate(builds, schedule_path, "bo_create_chain")
