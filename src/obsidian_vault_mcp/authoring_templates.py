"""Strict, model-neutral feature formulas and generated authoring context."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

import frontmatter

from . import bo_contract, config
from .vault import read_file

DEFAULTS = Path(__file__).parent / "templates"
VAULT_TEMPLATES = "Personal/Build Orchestrator/templates/"
CHECKLIST = "Personal/Build Orchestrator/rules/authoring-checklist.md"
VARIABLES = frozenset({
    "build_id", "review_id", "deploy_id", "feature_id", "title", "project",
    "what_this_means", "probes", "scope", "proof", "out_of_scope", "commit_prefix",
    "candidate_ref", "review_artifact", "summary_instruction", "deploy_canaries",
})
_TOKEN = re.compile(r"{{(.*?)}}", re.DOTALL)
_MODEL_LITERAL = re.compile(r"\b(?:gpt[- ]\d|claude[- ](?:sonnet|opus|haiku)|gemini[- ]\d|grok[- ]\d)", re.IGNORECASE)
_reported = set()
_cache_lock = threading.Lock()


class TemplateError(ValueError):
    pass


def registry_snapshot(timeout=2.0) -> dict:
    """Use BO's live registry resolver/validator, explicitly read-only.

    load(explicit_path) avoids registry auto-migration, reload audit and seed
    fallback writes. Only non-secret routing fields cross this subprocess.
    """
    directory = str(Path(bo_contract.resolve_authoring_contract_path()).parent)
    code = (
        "import sys,json;sys.path.insert(0,sys.argv[1]);import model_registry as m;"
        "d=m.load(m.registry_path());"
        "print(json.dumps({'version':d['version'],'pools':d['pools'],"
        "'models':[{k:x[k] for k in ('id','provider','roles','status')} for x in d['models']]}))"
    )
    try:
        process = subprocess.run([config.BO_AUTHORING_CONTRACT_PYTHON, "-c", code, directory],
                                 capture_output=True, text=True, timeout=timeout,
                                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        if process.returncode:
            raise TemplateError("live model registry is unavailable or invalid")
        return json.loads(process.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise TemplateError(f"live registry read failed: {exc}") from exc


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _validate(raw: str, kind: str, models: list[str]) -> dict:
    if len(raw.encode()) > 64_000:
        raise TemplateError("template exceeds 64 KB")
    try:
        document = frontmatter.loads(raw)
    except Exception as exc:
        raise TemplateError("malformed template frontmatter") from exc
    if document.metadata.get("kind") != kind or not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", str(document.metadata.get("version", ""))):
        raise TemplateError("template requires a kind and version")
    body = document.content
    if not body or "{%" in body or "{#" in body:
        raise TemplateError("template control blocks and filters are not supported")
    if _MODEL_LITERAL.search(raw) or any(re.search(r"(?<![\w-])" + re.escape(m) + r"(?![\w-])", raw, re.IGNORECASE) for m in models):
        raise TemplateError("templates must never name a model")
    names = []
    for match in _TOKEN.finditer(body):
        name = match.group(1).strip()
        if name not in VARIABLES:
            raise TemplateError(f"unknown template variable or filter: {name!r}")
        names.append(name)
    residue = _TOKEN.sub("", body)
    if "{{" in residue or "}}" in residue:
        raise TemplateError("unbalanced template variable")
    # These are the safety-bearing sections of the formula, derived by role.
    required = {"What this means", "Start gate", "Probes", "Scope", "Proof", "Out of scope", "Output", "Checkout write policy"}
    headings = set(re.findall(r"(?m)^## (.+)$", body))
    if not required <= headings or "summary_instruction" not in names:
        raise TemplateError("template omits required sections or summary instruction")
    if kind == "build" and ("candidate_ref" not in names or "commit_prefix" not in names):
        raise TemplateError("implementation template omits candidate or commit prefix")
    return {"version": str(document.metadata["version"]), "hash": _digest(raw),
            "body": body, "variables": sorted(set(names)), "raw": raw}


def load_template(name: str, kind: str, models: list[str], *, dry_run=False, version: str | None = None) -> tuple[dict, list[dict]]:
    cache = config.VAULT_PATH / ".bo-authoring-state" / "templates" / f"{name}.json"
    default = (DEFAULTS / f"{name}.md").read_text()
    warnings = []
    if version is not None:
        # Explicit replay uses immutable shipped bytes, never today's override
        # or last-known-good cache. Existing recorded hashes remain verifiable.
        if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", version) or version in {".", ".."}:
            raise TemplateError("invalid template version")
        current = _validate(default, kind, models)
        if version == current["version"]:
            return current, warnings
        try:
            archived = (DEFAULTS / "versions" / version / f"{name}.md").read_text()
        except FileNotFoundError as exc:
            raise TemplateError("unknown shipped template version") from exc
        template = _validate(archived, kind, models)
        if template["version"] != version:
            raise TemplateError("archived template version mismatch")
        return template, warnings
    try:
        raw, _ = read_file(VAULT_TEMPLATES + name + ".md")
    except FileNotFoundError:
        return _validate(default, kind, models), warnings
    try:
        template = _validate(raw, kind, models)
    except (ValueError, TypeError) as exc:
        with _cache_lock:
            try:
                cached = json.loads(cache.read_text())
                if cached["hash"] != _digest(cached["raw"]):
                    raise TemplateError("last-known-good template hash mismatch")
                template = _validate(cached["raw"], kind, models)
            except (OSError, ValueError, KeyError, TypeError):
                template = _validate(default, kind, models)
            identity = (str(config.VAULT_PATH), name, _digest(raw))
            if identity not in _reported:
                warnings.append({"code": "template_fallback", "template": name,
                                 "message": str(exc), "fallback_hash": template["hash"]})
                if not dry_run:
                    _reported.add(identity)
        return template, warnings
    if not dry_run:
        from .authoring_jobs import _write
        with _cache_lock:
            _write(cache, {"hash": template["hash"], "raw": raw})
    return template, warnings


def render(template: dict, variables: dict) -> str:
    def replace(match):
        name = match.group(1).strip()
        if name not in VARIABLES or name not in variables:
            raise TemplateError(f"unknown or missing template variable: {name}")
        return str(variables[name])
    # Single pass: caller text is data, never re-interpreted as template syntax.
    return _TOKEN.sub(replace, template["body"])


def _review_models(registry):
    active = {m["id"] for m in registry["models"] if m["status"] == "active" and "reviewer" in m["roles"]}
    pools = {name: [m for m in members if m in active]
             for name, members in registry["pools"].items() if name.endswith("review")}
    return pools


def feature_graph(feature: dict, *, dry_run=False):
    fid = feature["feature_id"]
    if not isinstance(fid, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", fid) or len(fid) > 180:
        raise TemplateError("feature_id must be a lowercase hyphenated id, at most 180 characters")
    project = feature["project"]
    if project not in ("build-orchestrator", "obsidian-web-mcp"):
        raise TemplateError("no default deploy formula for this project")
    for field in ("title", "what_this_means", "probes", "scope", "proof", "out_of_scope", "commit_prefix"):
        if not isinstance(feature.get(field), str) or not feature[field].strip():
            raise TemplateError(f"{field} must be non-empty text")
        if re.search(r'(?im)^ {0,3}#{1,6}[ \t]+checkout write policy(?:[ \t]+#+)?[ \t]*$', feature[field]):
            raise TemplateError(f'{field} contains an unsafe checkout policy heading')
    canaries = feature.get("deploy_canaries")
    if canaries is None:
        canaries = []
    if not isinstance(canaries, list) or len(canaries) > 50 or any(
        not isinstance(item, str) or not item.strip() or len(item) > 4000 for item in canaries
    ):
        raise TemplateError("deploy_canaries must be a list of up to 50 non-empty text checks, each at most 4000 characters")
    if any(re.search(r'(?im)^ {0,3}#{1,6}[ \t]+checkout write policy(?:[ \t]+#+)?[ \t]*$', item) for item in canaries):
        raise TemplateError("deploy_canaries contains an unsafe checkout policy heading")
    canary_block = ""
    if canaries:
        numbered = []
        for index, item in enumerate(canaries, 1):
            lines = item.strip().splitlines()
            numbered.append(f"   {index}. " + lines[0] + "".join("\n      " + line for line in lines[1:]))
        canary_block = "\n   Extra canaries (all must pass before publication):\n\n" + "\n".join(numbered) + "\n\n"
    registry = registry_snapshot()
    models = [m["id"] for m in registry["models"]]
    reviewers = _review_models(registry).get("frontier_review", [])
    if not reviewers:
        raise TemplateError("live registry has no active frontier reviewers")
    ids = [fid, fid + "-review", fid + "-deploy"]
    artifact = f"BS 2nd Brain/Alcove/Infrastructure/Hardening/Reviews/review-{fid}.md"
    variables = {k: feature[k] for k in VARIABLES if k in feature}
    variables.update(build_id=ids[0],review_id=ids[1],deploy_id=ids[2],
                     candidate_ref=f"refs/heads/candidate/{fid}",review_artifact=artifact,
                     deploy_canaries=canary_block)
    prefs = feature.get("model_preferences") or {}
    if not isinstance(prefs, dict) or set(prefs) - {"build", "review", "deploy"}:
        raise TemplateError("model_preferences must map build/review/deploy to routing fields")
    routing_keys = {"engine", "codex_model", "codex_reasoning_effort", "claude_model", "claude_effort", "cursor_model"}
    builds, provenance, warnings = [], {}, []
    for index, role in enumerate(("build", "review", "deploy")):
        name = role if role != "deploy" else f"deploy-{project}"
        template, notes = load_template(name, role, models, dry_run=dry_run)
        warnings.extend(notes)
        bid = ids[index]
        variables["summary_instruction"] = (
            f"Write a summary to /tmp/cc-summary-{bid}.txt where the FIRST LINE is exactly: {bid}.\n"
            "Then on following lines: status (pass/fail), files changed, test results."
        )
        preferred = prefs.get(role) or {}
        if not isinstance(preferred, dict) or set(preferred) - routing_keys:
            raise TemplateError("unknown model preference field")
        common = {"build_id": bid, "title": feature["title"] if index == 0 else role.title()+": "+feature["title"],
                  "project": project, "tier": "critical", "risk_domain": feature["risk_domain"],
                  "blast_radius": feature["blast_radius"], "reversible": feature["reversible"],
                  "shadowable": feature["shadowable"], "work_role": "reviewer" if role == "review" else "executor",
                  "depends_on": [] if index == 0 else [ids[index-1]],
                  "deployment_intent": {"build":"deferred","review":"not_applicable","deploy":"required"}[role],
                  "body_markdown": render(template, variables), **preferred}
        assertions = [{"type":"summary_valid"},{"type":"checkout_no_new_dirt"}]
        if role == "build":
            assertions.insert(0,{"type":"git_pushed","push_required":True,"head_matches":f"origin/candidate/{fid}"})
            common.update(resources=[], goal_id=fid, goal_root_build_id=fid, goal_completion=True)
        if role == "review":
            common.update(goal_id=bid, goal_root_build_id=bid, goal_completion=True)
        if role == "deploy":
            common["depends_on"] = []
            common["depends_on_goal"] = [{"producer_project": project, "goal_id": ids[1],
                                         "goal_root_build_id": ids[1], "evidence_class": "goal_completion"}]
            programs = (["orchestrator","orchestrator-dashboard","bs-control-session-bridge","log-syncer"]
                        if project == "build-orchestrator" else ["bs-brain-vault","cb-brain-vault","alcove-brain-vault"])
            assertions.extend({"type":"supervisor_program_running","program":p} for p in programs)
            assertions.append({"type":"deployment_status_ok"})
            if project == "build-orchestrator":
                assertions.extend({"type":"sha_deployed", "program":p, "head_matches":"origin/main"}
                                  for p in programs if p != "log-syncer")
            common["resources"] = [{"id":f"repo:{project}","mode":"exclusive"}] + [
                {"id":f"service:{p}","mode":"exclusive"} for p in programs]
            if project == "build-orchestrator":
                common["resources"].append({"id":"service:bs-control-mcp","mode":"exclusive"})
            # BO's review-goal gate verifies the achieving review's own result.
            # A fixed artifact_path cannot follow a generated re-review; the
            # deploy start gate additionally binds that approval to the ref head.
        common["completion_contract"] = {"assertions":assertions,"waivers": [] if role == "build" else [
            {"assertion":"git_pushed","reason":"Independent read-only review or deployment of an already-pushed exact candidate."}]}
        required = {'What this means', 'Start gate', 'Probes', 'Scope', 'Proof',
                    'Out of scope', 'Output', 'Checkout write policy'}
        headings = re.findall(r'(?m)^## (.+)$', common['body_markdown'])
        if not required <= set(headings) or headings.count('Checkout write policy') != 1:
            raise TemplateError('rendered formula omits required sections or has duplicate policy headings')
        builds.append(common)
        provenance[bid] = {"template_version":template["version"],"template_hash":template["hash"]}
    return builds, provenance, warnings


def bo_authoring_context(project: str) -> str:
    start = time.monotonic()
    try:
        if project not in ("build-orchestrator", "obsidian-web-mcp"):
            raise TemplateError("no default deploy formula for this project")
        registry = registry_snapshot(timeout=2)
        models = [m["id"] for m in registry["models"]]
        templates, warnings = {}, []
        sections = set()
        for role in ("build", "review", "deploy"):
            name = role if role != "deploy" else f"deploy-{project}"
            template, notes = load_template(name, role, models, dry_run=True)
            templates[name] = {"version":template["version"],"hash":template["hash"]}
            sections.update(re.findall(r"(?m)^## (.+)$",template["body"]))
            warnings.extend(notes)
        try:
            checklist, _ = read_file(CHECKLIST)
        except FileNotFoundError:
            checklist = None
        response = {"ok":True,"project":project,"templates":templates,
                    "registry_version":registry["version"],"reviewer_classes":_review_models(registry),
                    "allowed_models":[{"id":m["id"],"roles":m["roles"]} for m in registry["models"] if m["status"]=="active"],
                    "required_spec_sections":sorted(sections),"authoring_checklist_lines":checklist.splitlines() if checklist is not None else [],
                    "warnings":warnings}
        encoded = json.dumps(response,ensure_ascii=False,separators=(",",":"))
        if len(encoded.encode()) >= 8192 or time.monotonic()-start >= 5:
            return json.dumps({"ok":False,"code":"context_budget_exceeded","full_read_paths":[CHECKLIST,VAULT_TEMPLATES],
                               "error":"Full generated context exceeds its budget; read the source documents."})
        return encoded
    except (ValueError, OSError, bo_contract.BOContractError) as exc:
        return json.dumps({"ok":False,"error":str(exc)})
