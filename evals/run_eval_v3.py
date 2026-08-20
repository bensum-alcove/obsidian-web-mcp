#!/usr/bin/env python3
"""run_eval_v3.py -- frozen v3 retrieval benchmark for the Obsidian vault MCP tools.

Deliberately separate from run_eval.py / eval-set.yaml (v2): v2's own corpus,
history/, and weekly cron invocation are UNCHANGED by this file -- v3 is an
additive, capability-based benchmark (known facts, current-vs-stale,
canonical/contradiction precedence, entities/aliases, exact identifiers,
paraphrase, cross-document, negative/unknown, recent-update freshness,
archive isolation) meant to become the new frozen benchmark that governs
future retrieval-ranking changes. See evals/eval-set-v3.yaml's header comment
and `Personal/Build Orchestrator/specs/vault-retrieval-eval-v3.md`.

Reuses run_eval.py's live-tool-import machinery and eval-artifact exclusion
list by loading it as a module (importlib, by file path -- evals/ is not an
installed package) rather than duplicating that logic; v3 adds its own
exclusion globs on top of v2's list for the duration of this process only.

Cron: intended to run alongside (not instead of) the existing v2 weekly job,
same LIVE_SRC_ROOT / VAULT_NAME / VAULT_PATH conventions as run_eval.py.

Vault writes: exactly one dated report file, directly to the vault
filesystem, under its own report directory (retrieval-eval-v3), kept
separate from v2's `retrieval-eval/` reports. Run-over-run history and the
frozen corpus hash / deploy gates live in this repo's evals/history-v3/, not
in the vault.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_V3_PATH = REPO_ROOT / "evals" / "eval-set-v3.yaml"
HISTORY_V3_DIR = REPO_ROOT / "evals" / "history-v3"
CORPUS_HASH_PATH = HISTORY_V3_DIR / "CORPUS_HASH"
DEPLOY_GATES_PATH = HISTORY_V3_DIR / "deploy-gates.yaml"

CORPUS_VERSION = "v3.0.0"
VAULT_NAME = __import__("os").environ.get("VAULT_NAME", "bs-brain")
REPORT_DIR_IN_VAULT = "BS 2nd Brain/Alcove/Infrastructure/retrieval-eval-v3"

TOP_N = 15        # matches v2's over-fetch depth -- same exclusion-safety rationale
R_AT_K = 5        # R@5
UNDERPOWERED_THRESHOLD = 5   # category n below this is flagged, never silently averaged in as equal-weight

RUBRIC_CATEGORIES = {
    "current_vs_stale",
    "canonical_contradiction_precedence",
    "recent_update_freshness",
    "archive_isolation",
}


def _load_v2_module():
    """Load run_eval.py (v2) as a module by path -- reuses its live-tool-import
    and exclusion-glob machinery without touching v2's own file, corpus, or
    history. v2 continues to run completely independently of this script."""
    spec = importlib.util.spec_from_file_location("run_eval_v2", REPO_ROOT / "evals" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_v2 = _load_v2_module()

# v3-specific additions to v2's exclusion glob list -- this build's own vault
# artifacts (v3 report dir, its own spec, its own build-log output) must not
# contaminate scoring, same rationale as v2's EVAL_EXCLUDE_GLOBS. Extending
# v2's list in place (rather than copying it) means _v2._is_eval_excluded /
# _v2._extract_paths automatically honour the combined list for the lifetime
# of this process -- v2's own file on disk is never touched.
_v2.EVAL_EXCLUDE_GLOBS.extend([
    f"{REPORT_DIR_IN_VAULT}/**",
    "BS 2nd Brain/Alcove/Infrastructure/Build Logs/vault-retrieval-eval-v3-output.md",
    "Personal/Build Orchestrator/specs/vault-retrieval-eval-v3.md",
])


def load_eval_set_v3() -> tuple[list[dict], str]:
    raw = EVAL_SET_V3_PATH.read_text(encoding="utf-8")
    entries = yaml.safe_load(raw)
    for entry in entries:
        entry.setdefault("expected_paths", [])
        entry.setdefault("decoy_paths", [])
        entry.setdefault("negative", False)
        entry.setdefault("require_all", False)
        if entry["negative"] and entry["expected_paths"]:
            raise ValueError(f"negative entry must have empty expected_paths: {entry['question']!r}")
        if not entry["negative"] and not entry["expected_paths"]:
            raise ValueError(f"non-negative entry missing expected_paths: {entry['question']!r}")
    return entries, raw


def compute_corpus_hash(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def stratify_categories(eval_set: list[dict]) -> dict[str, dict]:
    counts = Counter(entry["category"] for entry in eval_set)
    return {
        category: {"n": n, "underpowered": n < UNDERPOWERED_THRESHOLD}
        for category, n in sorted(counts.items())
    }


def _validate_no_excluded_paths_v3(eval_set: list[dict], vault_path: Path) -> None:
    """Contamination guard: neither an expected_path nor a decoy_path may be
    something the runner itself excludes from scoring -- that would make the
    question unwinnable (expected) or the decoy silently untestable (decoy)."""
    violations = []
    for entry in eval_set:
        for path in entry["expected_paths"] + entry["decoy_paths"]:
            if _v2._is_eval_excluded(path, vault_path):
                violations.append((entry["category"], entry["question"], path))
    if violations:
        print("[error] expected/decoy path excluded from scoring -- contamination guard tripped:", file=sys.stderr)
        for category, question, path in violations:
            print(f"  [{category}] {question!r} -> excluded path: {path}", file=sys.stderr)
        sys.exit(1)


def _needs_rubric(entry: dict) -> bool:
    return (
        not entry["negative"]
        and not entry["decoy_paths"]
        and entry["category"] in RUBRIC_CATEGORIES
    )


def _rubric_text(entry: dict) -> str:
    note = (entry.get("category_note") or "").strip()
    return (
        "No decoy defined for deterministic source/freshness scoring on this "
        "question -- manually confirm the tool's top-ranked result actually "
        f"satisfies the intent below, not just a keyword/topic match. {note}"
    ).strip()


def score_entry(returned_paths: list[str], entry: dict) -> dict:
    """Per-query, per-tool score: standard R@5/MRR (skipped for negative
    entries), plus deterministic source-correctness (needs decoy_paths) and,
    for require_all cross_document entries, a synthesis_hit."""
    expected = entry["expected_paths"]
    decoys = entry["decoy_paths"]
    top_k = returned_paths[:R_AT_K]
    result: dict = {"hit": None, "rr": None, "source_correct": None, "synthesis_hit": None}

    if entry["negative"]:
        if decoys:
            result["source_correct"] = 0.0 if any(p in decoys for p in top_k) else 1.0
        return result

    result["hit"] = 1.0 if any(p in expected for p in top_k) else 0.0
    rr = 0.0
    for rank, path in enumerate(returned_paths, start=1):
        if path in expected:
            rr = 1.0 / rank
            break
    result["rr"] = rr

    if entry["require_all"]:
        result["synthesis_hit"] = 1.0 if all(p in top_k for p in expected) else 0.0

    if decoys:
        for path in returned_paths:
            if path in expected:
                result["source_correct"] = 1.0
                break
            if path in decoys:
                result["source_correct"] = 0.0
                break

    return result


def run_tool_eval_v3(tool_fn, kwargs_fn, eval_set: list[dict], vault_path: Path) -> dict:
    per_category: dict[str, list[dict]] = {}
    rubric_flagged: list[dict] = []

    for entry in eval_set:
        raw = tool_fn(entry["question"], **kwargs_fn())
        paths = _v2._extract_paths(raw, vault_path)
        scored = score_entry(paths, entry)
        per_category.setdefault(entry["category"], []).append(scored)

        if _needs_rubric(entry):
            rubric_flagged.append({
                "category": entry["category"],
                "question": entry["question"],
                "rubric": _rubric_text(entry),
                "top_result": paths[0] if paths else None,
            })

    def _agg(scores: list[dict]) -> dict:
        hits = [s["hit"] for s in scores if s["hit"] is not None]
        rrs = [s["rr"] for s in scores if s["rr"] is not None]
        source = [s["source_correct"] for s in scores if s["source_correct"] is not None]
        synth = [s["synthesis_hit"] for s in scores if s["synthesis_hit"] is not None]
        out = {"n": len(scores)}
        if hits:
            out["r_at_5"] = round(statistics.mean(hits), 4)
            out["mrr"] = round(statistics.mean(rrs), 4)
        if source:
            out["source_correctness"] = round(statistics.mean(source), 4)
            out["source_correctness_n"] = len(source)
        if synth:
            out["synthesis_hit_rate"] = round(statistics.mean(synth), 4)
            out["synthesis_hit_n"] = len(synth)
        return out

    summary: dict = {}
    all_scores: list[dict] = []
    for category, scores in per_category.items():
        summary[category] = _agg(scores)
        all_scores.extend(scores)
    summary["overall"] = _agg(all_scores)
    summary["_rubric_flagged"] = rubric_flagged
    return summary


def load_previous_history() -> dict | None:
    if not HISTORY_V3_DIR.is_dir():
        return None
    files = sorted(p for p in HISTORY_V3_DIR.glob("*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def save_history_v3(date_str: str, results: dict) -> None:
    HISTORY_V3_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_V3_DIR / f"{date_str}.json", "w") as f:
        json.dump(results, f, indent=2)


def check_or_freeze_corpus_hash(corpus_hash: str, freeze: bool) -> dict:
    """Returns {"frozen_hash": str|None, "drift": bool, "just_frozen": bool}."""
    HISTORY_V3_DIR.mkdir(parents=True, exist_ok=True)
    existing = None
    if CORPUS_HASH_PATH.is_file():
        for line in CORPUS_HASH_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("sha256:"):
                existing = line.split("sha256:", 1)[1].strip()

    if freeze or existing is None:
        CORPUS_HASH_PATH.write_text(
            f"corpus_version: {CORPUS_VERSION}\n"
            f"sha256: {corpus_hash}\n"
            f"frozen_at: '{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}'\n",
            encoding="utf-8",
        )
        return {"frozen_hash": corpus_hash, "drift": False, "just_frozen": True}

    drift = existing != corpus_hash
    if drift:
        print(
            f"[warn] corpus hash drift: frozen={existing} current={corpus_hash} -- "
            "eval-set-v3.yaml changed since the last freeze. If deliberate, bump "
            "corpus_version in the yaml header and re-run with --freeze-corpus.",
            file=sys.stderr,
        )
    return {"frozen_hash": existing, "drift": drift, "just_frozen": False}


def write_deploy_gates(results: dict, date_str: str) -> None:
    """Pre-register per-category, per-tool deploy-gate floors from this
    baseline run -- calibration floors, not yet tuned against multiple runs.
    A future ranking-code change should not regress a category's R@5/MRR
    below its floor without an explicit, reviewed justification."""
    gates: dict = {}
    for tool, summary in results.items():
        tool_gates = {}
        for category, scores in summary.items():
            if category == "_rubric_flagged":
                continue
            if "r_at_5" not in scores:
                continue
            tool_gates[category] = {
                "floor_r_at_5": scores["r_at_5"],
                "floor_mrr": scores["mrr"],
                "n": scores["n"],
            }
        gates[tool] = tool_gates

    payload = {
        "corpus_version": CORPUS_VERSION,
        "baseline_date": date_str,
        "note": (
            "Pre-registered from the first v3 baseline run, before any "
            "ranking-code change. Calibration floors, not yet tuned against "
            "multiple runs -- review before treating as a hard CI gate. A "
            "future ranking change should not regress any category below its "
            "floor_r_at_5/floor_mrr without an explicit, reviewed justification."
        ),
        "gates": gates,
    }
    with open(DEPLOY_GATES_PATH, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _fmt_delta(current: float, previous: float | None) -> str:
    if previous is None:
        return "baseline"
    delta = current - previous
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def render_report_v3(
    date_str: str,
    results: dict,
    previous: dict | None,
    stratification: dict,
    hash_status: dict,
) -> str:
    tools = list(results.keys())
    lines = [
        "---",
        "tags:",
        "  - infrastructure",
        "  - vault-mcp",
        "  - evaluation",
        "  - retrieval-eval",
        "  - retrieval-eval-v3",
        "type: reference",
        "build_id: vault-retrieval-eval-v3",
        f"corpus_version: {CORPUS_VERSION}",
        f"updated: '{date_str}'",
        "---",
        "",
        f"# Retrieval Eval Report v3 — {date_str}",
        "",
        "## What this means",
        "",
        "Frozen v3 benchmark: ten capability-based categories (known facts, "
        "current-vs-stale, canonical/contradiction precedence, entities/aliases, "
        "exact identifiers, paraphrase, cross-document, negative/unknown, "
        "recent-update freshness, archive isolation) scored against every deployed "
        "vault retrieval tool. This is separate from and does not replace the v2 "
        "`retrieval-eval/` weekly report — v2 keeps running for its own time series. "
        "R@5/MRR are standard; `source_correctness` and `synthesis_hit_rate` are new "
        "answer-level metrics (see `evals/eval-set-v3.yaml` header for definitions).",
        "",
        f"**Corpus hash:** `sha256:{hash_status['frozen_hash']}`"
        + (" (just frozen this run)" if hash_status["just_frozen"] else "")
        + (" **DRIFTED from frozen baseline**" if hash_status["drift"] else ""),
        "",
        "---",
        "",
        "## Category stratification",
        "",
        "| Category | n | Underpowered? |",
        "|---|---|---|",
    ]
    for category, info in stratification.items():
        flag = "**YES**" if info["underpowered"] else "no"
        lines.append(f"| {category} | {info['n']} | {flag} |")

    lines += ["", "## Overall", "", "| Tool | R@5 | MRR | Δ R@5 | Δ MRR | n |", "|------|-----|-----|-------|-------|---|"]
    for tool in tools:
        overall = results[tool]["overall"]
        prev_overall = (previous or {}).get(tool, {}).get("overall")
        d_r5 = _fmt_delta(overall["r_at_5"], prev_overall["r_at_5"] if prev_overall else None)
        d_mrr = _fmt_delta(overall["mrr"], prev_overall["mrr"] if prev_overall else None)
        lines.append(f"| {tool} | {overall['r_at_5']:.4f} | {overall['mrr']:.4f} | {d_r5} | {d_mrr} | {overall['n']} |")

    lines += ["", "## By category", ""]
    categories = sorted({c for tool in tools for c in results[tool] if c not in ("overall", "_rubric_flagged")})
    for category in categories:
        lines += [f"### {category}", "", "| Tool | R@5 | MRR | Source-correct | Synthesis hit | n |", "|------|-----|-----|-----------------|----------------|---|"]
        for tool in tools:
            cat_scores = results[tool].get(category)
            if not cat_scores:
                continue
            r5 = f"{cat_scores['r_at_5']:.4f}" if "r_at_5" in cat_scores else "n/a"
            mrr = f"{cat_scores['mrr']:.4f}" if "mrr" in cat_scores else "n/a"
            src = f"{cat_scores['source_correctness']:.4f} (n={cat_scores['source_correctness_n']})" if "source_correctness" in cat_scores else "no decoy defined"
            synth = f"{cat_scores['synthesis_hit_rate']:.4f} (n={cat_scores['synthesis_hit_n']})" if "synthesis_hit_rate" in cat_scores else "n/a"
            lines.append(f"| {tool} | {r5} | {mrr} | {src} | {synth} | {cat_scores['n']} |")
        lines.append("")

    lines += ["## Manual rubric review needed (no deterministic decoy defined)", ""]
    any_rubric = False
    for tool in tools:
        flagged = results[tool].get("_rubric_flagged", [])
        if not flagged:
            continue
        any_rubric = True
        lines.append(f"### {tool}")
        lines.append("")
        for item in flagged:
            lines.append(f"- **[{item['category']}]** {item['question']!r} — top result: `{item['top_result']}`")
            lines.append(f"  - {item['rubric']}")
        lines.append("")
    if not any_rubric:
        lines.append("_None this run._")
        lines.append("")

    if previous is None:
        lines.append("_No prior v3 run found — this is the baseline._")

    return "\n".join(lines) + "\n"


def main() -> None:
    freeze = "--freeze-corpus" in sys.argv

    config, src_root = _v2._load_live_tools()
    print(f"Tool source: {src_root}", file=sys.stderr)
    print(f"VAULT_PATH: {config.VAULT_PATH}", file=sys.stderr)

    eval_set, raw_text = load_eval_set_v3()
    corpus_hash = compute_corpus_hash(raw_text)
    stratification = stratify_categories(eval_set)
    for category, info in stratification.items():
        if info["underpowered"]:
            print(f"[warn] underpowered category: {category} (n={info['n']} < {UNDERPOWERED_THRESHOLD})", file=sys.stderr)

    _validate_no_excluded_paths_v3(eval_set, config.VAULT_PATH)

    runners = _v2.build_tool_runners(config, src_root)
    if not runners:
        print("[error] no retrieval tools importable -- aborting", file=sys.stderr)
        sys.exit(1)

    results = {}
    for tool_name, (tool_fn, kwargs_fn) in runners.items():
        print(f"Scoring {tool_name} (v3)...", file=sys.stderr)
        results[tool_name] = run_tool_eval_v3(tool_fn, kwargs_fn, eval_set, config.VAULT_PATH)

    hash_status = check_or_freeze_corpus_hash(corpus_hash, freeze)

    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    previous = load_previous_history()
    save_history_v3(date_str, results)

    if freeze or not DEPLOY_GATES_PATH.is_file():
        write_deploy_gates(results, date_str)

    report = render_report_v3(date_str, results, previous, stratification, hash_status)
    report_path = config.VAULT_PATH / REPORT_DIR_IN_VAULT / f"{date_str}-report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"Report written to {report_path}", file=sys.stderr)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
