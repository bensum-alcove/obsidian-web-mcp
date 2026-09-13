"""Build Orchestrator authoring tools (vault-bo-authoring-mcp-v1).

bo_validate_build_graph / bo_create_build / bo_create_chain /
bo_activate_existing_spec. All four delegate BO schema validation and
rendering entirely to the authoring contract adapter (bo_contract.py ->
authoring_contract.py's JSON CLI) -- this module only:

  - shapes typed tool inputs into the adapter's node/spec JSON shape,
  - constructs one valid YAML schedule document (never string-splices an
    entry after a flow-style empty `builds: []`),
  - enforces the write ordering invariant (validate the whole graph -> write
    every new spec -> write/replace the schedule last, as the activation
    boundary), so a pre-schedule failure leaves any already-written spec
    orphaned (inert -- not referenced by any schedule entry) rather than
    producing a half-activated graph,
  - fails closed if the adapter is unavailable or a schema-version mismatch is
    detected (bo_contract.BOContractError) -- never guesses at validity.

Create paths append to an EXISTING schedule file, never created fresh.
bo_activate_existing_spec reuses the same prepare/compose/commit primitive
to bind one already-written inert spec as the final schedule write only.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import frontmatter
import yaml

from .. import bo_contract
from ..bo_guard import SPECS_PREFIX, schedule_builds_from_content
from ..frontmatter_safe import update_frontmatter_field
from ..vault import RevisionConflictError, conflict_payload, read_file, write_file_atomic

logger = logging.getLogger(__name__)

REQUIRED_BUILD_FIELDS = ("build_id", "title", "body_markdown", "tier", "project")
_BUILDS_KEY_RE = re.compile(r"^(builds\s*:)", re.MULTILINE)
_SPEC_HEADING_RE = re.compile(r"^#\s+(?:.+?\s+[—-]\s+)?(.+)\s*$", re.MULTILINE)


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
    for optional in ("risk_domain", "blast_radius", "reversible", "shadowable", "capital_path",
                      "engine", "notes", "resources"):
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
    for optional in ("risk_domain", "blast_radius", "reversible", "shadowable", "capital_path",
                      "program", "tags", "created", "completion_contract", "deployment_intent",
                      "resources", "review_gate"):
        if build.get(optional) is not None:
            item[optional] = build[optional]
    return item


def _splice_schedule_entry_field(entry_text: str, field_name: str, value) -> str:
    """Add one extra key to an already-rendered `builds:` entry fragment.

    render_schedule_entry()'s own key allowlist (_SCHEDULE_ENTRY_KEY_ORDER)
    doesn't include `resources` yet, so it's dropped from the rendered text
    even though it's already present -- and validated -- on the raw
    schedule_entry dict every validate_graph call receives. This inverts
    render_schedule_entry's own "builds: [entry]" dump/2-space-indent
    convention just far enough to reparse the fragment, add the key, and
    reapply the identical convention -- a text-layout mirror, not a new BO
    validation rule; the adapter's own validate_graph is still the only
    thing that decides whether the resulting shape is valid.
    """
    unindented = []
    for line in entry_text.splitlines():
        if line.startswith("  - "):
            unindented.append("- " + line[4:])
        elif line.startswith("  "):
            unindented.append(line[2:])
        else:
            unindented.append(line)
    parsed = yaml.safe_load("builds:\n" + "\n".join(unindented) + "\n")
    parsed["builds"][0][field_name] = value
    dumped = yaml.safe_dump(parsed, default_flow_style=False, sort_keys=False, allow_unicode=True)
    out = []
    for line in dumped.split("\n")[1:]:
        if not line.strip():
            continue
        out.append(("  - " + line[2:]) if line.startswith("- ") else ("  " + line))
    return "\n".join(out) + "\n"


def _parse_existing_schedule(content: str, source_name: str) -> tuple[dict, list]:
    """Parse one on-disk schedule through the canonical adapter parser.

    Distinguishes malformed YAML (BOToolError, fail closed) from an empty
    `builds` sequence (`[]`, null, or missing), which is a valid empty
    document that the shared composer must rewrite as a block sequence.
    """
    try:
        parsed = bo_contract.parse_schedule_document(content, source_name=source_name)
    except ValueError as e:
        raise BOToolError(f"malformed schedule {source_name}: {e}") from e
    if not isinstance(parsed, dict):
        raise BOToolError(f"malformed schedule {source_name}: document is not a mapping")
    builds = parsed.get("builds")
    if builds is None:
        builds = []
    elif not isinstance(builds, list):
        raise BOToolError(f"malformed schedule {source_name}: builds is not a list")
    return parsed, builds


def compose_schedule_document(existing_content: str, new_entry_texts: list[str], *,
                              source_name: str, new_ids: list[str] | None = None) -> str:
    """Build one valid YAML schedule document containing the new entries.

    Shared by bo_create_build, bo_create_chain, and bo_activate_existing_spec.
    Empty sequences (`builds: []`, `builds:` with a null/missing value, or a
    missing `builds:` key) are rewritten as a block `builds:` sequence --
    never string-spliced after a flow-style empty list. Non-empty documents
    keep existing bytes and append the new rendered entry fragments.
    """
    if not new_entry_texts:
        raise BOToolError(f"compose_schedule_document requires at least one new entry for {source_name}")

    _parsed, existing_builds = _parse_existing_schedule(existing_content, source_name)
    existing_ids = {entry.get("id") for entry in existing_builds if isinstance(entry, dict)}
    if new_ids:
        overlap = existing_ids & set(new_ids)
        if overlap:
            raise BOToolError(
                f"refusing to duplicate already-bound build id(s) in {source_name}: {sorted(overlap)}"
            )

    fragments = [text.rstrip("\n") for text in new_entry_texts]
    entry_block = "\n\n".join(fragments) + "\n"

    if existing_builds:
        return existing_content.rstrip("\n") + "\n\n" + entry_block

    match = _BUILDS_KEY_RE.search(existing_content)
    if match is None:
        prefix = existing_content.rstrip("\n")
        joiner = "\n\n" if prefix else ""
        return f"{prefix}{joiner}builds:\n\n{entry_block}"
    prefix = existing_content[: match.start()]
    return prefix + "builds:\n\n" + entry_block


def _build_from_existing_spec(build_id: str, spec_markdown: str, spec_path: str) -> dict:
    """Derive a typed build dict from an already-written spec.

    Schedule-entry rendering still goes through the adapter via
    `_prepare_graph`; this only copies identity/body fields the adapter
    already understands. Refuses identity mismatch and a spec that cannot
    supply the fields `_normalize_build` requires.
    """
    parsed = frontmatter.loads(spec_markdown)
    fm = parsed.metadata or {}
    body = (parsed.content or "").strip()
    spec_build_id = fm.get("build_id")
    if spec_build_id and spec_build_id != build_id:
        raise BOToolError(
            f"ambiguous spec identity: spec build_id {spec_build_id!r} != requested {build_id!r}"
        )
    stem = Path(spec_path).stem
    if stem != build_id:
        raise BOToolError(
            f"ambiguous spec identity: spec filename stem {stem!r} != requested {build_id!r}"
        )
    title = fm.get("title")
    if not title:
        heading = _SPEC_HEADING_RE.search(parsed.content or "")
        title = heading.group(1).strip() if heading else None
    if not title:
        raise BOToolError(f"existing spec at {spec_path!r} is missing a title")
    if not body:
        raise BOToolError(f"existing spec at {spec_path!r} is missing a body")
    for required in ("tier", "project"):
        if not fm.get(required):
            raise BOToolError(f"existing spec at {spec_path!r} is missing required field {required!r}")
    build = {
        "build_id": build_id,
        "title": title,
        "body_markdown": body,
        "tier": fm["tier"],
        "project": fm["project"],
        "status": fm.get("status") or "ready",
        "depends_on": fm.get("depends_on") or [],
    }
    for optional in (
        "description", "run_when", "risk_domain", "blast_radius", "reversible", "shadowable",
        "capital_path", "engine", "notes", "program", "tags", "created", "completion_contract",
        "deployment_intent", "resources", "review_gate",
    ):
        if fm.get(optional) is not None:
            build[optional] = fm[optional]
    return build


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
    Malformed existing content fails closed (BOToolError) rather than being
    treated as an empty graph.
    """
    try:
        content, _ = read_file(schedule_path)
    except FileNotFoundError:
        return [], None

    _parsed, existing_builds = _parse_existing_schedule(content, schedule_path)

    project = None
    try:
        project = frontmatter.loads(content).metadata.get("project")
    except Exception:
        pass

    nodes = []
    for entry in existing_builds:
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


def _prepare_graph(builds: list[dict], schedule_path: str, mode: str,
                   spec_markdown_overrides: dict[str, str] | None = None) -> dict:
    """Shape + render + validate a proposed graph. Never writes anything.

    Validates the COMPLETE resulting graph -- every entry already in
    schedule_path plus the newly-proposed ones -- not just the newly-proposed
    rows in isolation (B2). `mode` (and its strict-new-only checks) applies
    only to the newly-proposed build ids; pre-existing entries are still
    included in every cross-node check (duplicate id, mixed project,
    dependency/cycle) but evaluated leniently at the per-node level, via
    `bo_contract.validate_graph`'s `new_ids` parameter.

    `spec_markdown_overrides` (used by bo_activate_existing_spec) replaces the
    adapter-rendered spec body with the already-written on-disk spec so
    validation inspects the artifact that will actually be activated, not a
    re-render. Schedule-entry YAML still comes from the adapter renderer.

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

    # `resources` and `review_gate` are current spec-frontmatter fields the
    # adapter's own validate_spec_frontmatter already understands (it runs
    # resource_resolve.validate_resource_claims_shape / _validate_review_gate_block
    # against them), but render_spec() doesn't yet accept them as render
    # parameters, so the adapter's own render_graph silently drops them. Splice
    # them into the rendered frontmatter with the same generic, semantics-free
    # YAML field-set helper VaultWriteInput.merge_frontmatter uses -- this adds
    # no BO validation logic locally; the spliced result is what validate_graph
    # below (and the eventual write) actually sees, so the adapter still has
    # the only word on whether the shape is valid.
    overrides = spec_markdown_overrides or {}
    for b in normalized_builds:
        if b["build_id"] in overrides:
            rendered[b["build_id"]]["spec"] = overrides[b["build_id"]]
        else:
            spec_text = rendered[b["build_id"]]["spec"]
            for field_name in ("resources", "review_gate"):
                if b.get(field_name) is not None:
                    spec_text = update_frontmatter_field(spec_text, field_name, b[field_name], require_existing=False)
            rendered[b["build_id"]]["spec"] = spec_text

        # `resources` also belongs on the schedule entry itself (the resource-
        # aware scheduler reads locks from the schedule file, not the spec) --
        # same rendering gap, same fix, at the schedule_entry fragment layer.
        if b.get("resources") is not None:
            rendered[b["build_id"]]["schedule_entry"] = _splice_schedule_entry_field(
                rendered[b["build_id"]]["schedule_entry"], "resources", b["resources"]
            )

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


def _activate(builds: list[dict], schedule_path: str, tool_name: str, *,
              write_new_specs: bool = True,
              spec_markdown_overrides: dict[str, str] | None = None) -> str:
    """Shared create/activate path for bo_create_build, bo_create_chain, and
    bo_activate_existing_spec.

    Always validates in "strict_new" mode -- `compat_existing` is a read-only
    audit/compatibility mode for the historical corpus and must not be
    caller-selectable on a path that writes new artifacts (B3: "compatibility
    mode is exposed on mutation tools ... allows new malformed artifacts").
    Schedule-document construction always goes through compose_schedule_document.
    When write_new_specs is False, specs must already exist and only the
    schedule activation write is performed.
    """
    try:
        prep = _prepare_graph(
            builds, schedule_path, "strict_new",
            spec_markdown_overrides=spec_markdown_overrides,
        )
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
    # Activation of an already-written spec inverts this: the spec must exist
    # and is never rewritten.
    if write_new_specs:
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

    new_ids = [node["build_id"] for node in prep["nodes"]]
    new_entry_texts = [prep["rendered"][node["build_id"]]["schedule_entry"] for node in prep["nodes"]]
    try:
        new_schedule_content = compose_schedule_document(
            schedule_content, new_entry_texts, source_name=schedule_path, new_ids=new_ids,
        )
    except BOToolError as e:
        return json.dumps({"ok": False, "error": str(e), "activated": False})

    # Validate the fully-rendered result BEFORE writing anything -- a
    # half-written pair (spec written, schedule broken) is worse than writing
    # neither. Structural self-check only (parses, every new id present, no
    # duplicate ids) -- not a second schema validator; BO semantics were
    # already checked above.
    try:
        parsed_builds = schedule_builds_from_content(new_schedule_content, source_name=schedule_path)
    except bo_contract.BOContractError as e:
        return json.dumps({"ok": False, "error": str(e), "code": e.code, "activated": False})
    if parsed_builds is None:
        return json.dumps({
            "ok": False,
            "error": "generated schedule content failed to re-parse before write -- refusing to write anything",
            "activated": False,
        })
    parsed_id_list = [b.get("id") for b in parsed_builds if isinstance(b, dict)]
    if len(parsed_id_list) != len(set(parsed_id_list)):
        return json.dumps({
            "ok": False,
            "error": "generated schedule content would duplicate a build id -- refusing to write anything",
            "activated": False,
        })
    parsed_ids = set(parsed_id_list)
    for node in prep["nodes"]:
        if node["build_id"] not in parsed_ids:
            return json.dumps({
                "ok": False,
                "error": f"generated schedule content has no entry for {node['build_id']!r} after append",
                "activated": False,
            })

    created = []
    try:
        if write_new_specs:
            for node in prep["nodes"]:
                spec_path = node["schedule_entry"]["spec_path"]
                write_file_atomic(spec_path, node["spec_markdown"], create_dirs=True, tool=tool_name)
                created.append({
                    "build_id": node["build_id"],
                    "spec_path": spec_path,
                    "project": node["schedule_entry"].get("project"),
                    "depends_on": node["schedule_entry"].get("depends_on", []),
                })
        else:
            created = [
                {
                    "build_id": node["build_id"],
                    "spec_path": node["schedule_entry"]["spec_path"],
                    "project": node["schedule_entry"].get("project"),
                    "depends_on": node["schedule_entry"].get("depends_on", []),
                }
                for node in prep["nodes"]
            ]

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
            "orphaned_specs": [c["spec_path"] for c in created] if write_new_specs else [],
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


def bo_activate_existing_spec(build_id: str, schedule_path: str, spec_path: str | None = None) -> str:
    """Activate one already-written inert spec onto an existing schedule.

    Reuses `_activate` / `compose_schedule_document` -- no second state machine.
    Derives the schedule entry from the spec, validates the full graph
    strict_new, and performs only the final schedule write. Refuses missing
    or ambiguous specs, already-bound ids, a missing schedule, and any path
    that would rewrite the spec or accept raw YAML.
    """
    if not build_id or not isinstance(build_id, str):
        return json.dumps({"ok": False, "error": "build_id is required", "activated": False})
    canonical = _spec_path_for(build_id)
    if spec_path and spec_path != canonical:
        return json.dumps({
            "ok": False,
            "error": (
                f"ambiguous spec identity: spec_path {spec_path!r} does not match "
                f"canonical path {canonical!r} for build_id {build_id!r}"
            ),
            "activated": False,
        })
    resolved_spec_path = canonical
    try:
        spec_markdown, _ = read_file(resolved_spec_path)
    except FileNotFoundError:
        return json.dumps({
            "ok": False,
            "error": f"missing spec at {resolved_spec_path!r}",
            "activated": False,
        })
    try:
        build = _build_from_existing_spec(build_id, spec_markdown, resolved_spec_path)
    except BOToolError as e:
        return json.dumps({"ok": False, "error": str(e), "activated": False})
    return _activate(
        [build], schedule_path, "bo_activate_existing_spec",
        write_new_specs=False,
        spec_markdown_overrides={build_id: spec_markdown},
    )
