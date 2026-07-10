#!/usr/bin/env python3
"""run_eval.py — R@5 / MRR retrieval eval runner for the Obsidian vault MCP tools.

Cron: weekly, Sundays 16:00 UTC, one invocation per VAULT_PATH / VAULT_NAME.
Scores vault_search, vault_semantic_search, and vault_query (if importable) against
evals/eval-set.yaml using direct library import of the tool implementations.

Tool code is imported from the LIVE deployed checkout (feature/vault-tools-v2,
see SYSTEM-FACTS.md "Repo Checkouts" section), not this repo's own src/ — this repo
(main, dev copy) is what cron scripts live in, but the live checkout is what's
actually serving vault_search/vault_semantic_search/vault_query traffic, and only
the live checkout has vault_query. Override via LIVE_SRC_ROOT if that path moves.

Vault writes: this script writes exactly one file — the dated report — directly
to the vault filesystem (same-box process, no HTTP/MCP needed per SYSTEM-FACTS.md).
Run-over-run score history for delta computation is kept in this repo's
evals/history/, not in the vault.
"""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = REPO_ROOT / "evals" / "eval-set.yaml"
HISTORY_DIR = REPO_ROOT / "evals" / "history"

LIVE_SRC_ROOT = Path(
    __import__("os").environ.get(
        "LIVE_SRC_ROOT", "/mnt/c/Users/Ben Sum/obsidian-web-mcp/src"
    )
)

VAULT_NAME = __import__("os").environ.get("VAULT_NAME", "bs-brain")
REPORT_DIR_IN_VAULT = "BS 2nd Brain/Alcove/Infrastructure/retrieval-eval"

# eval-set.md (human-readable copy) lives in the vault and contains every question
# verbatim — it would otherwise self-match as a literal substring hit on every query
# and win vault_search's keyword-fallback leg. Exclude the eval's own artifacts from
# scoring; they aren't real content.
_SELF_REFERENTIAL_PREFIX = REPORT_DIR_IN_VAULT + "/"

TOP_N = 10        # results requested per tool call (MRR can rank beyond 5)
R_AT_K = 5        # R@5


def _load_live_tools():
    """Import config + tool modules from the live deployed checkout.

    Falls back to this repo's own src/ (dev copy) if the live checkout is
    unreachable, so the eval still runs (minus vault_query) rather than failing.
    """
    src_root = LIVE_SRC_ROOT if LIVE_SRC_ROOT.is_dir() else REPO_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from obsidian_vault_mcp import config  # noqa: E402

    # eval-set.md and every dated report live under retrieval-eval/ and contain every
    # question verbatim — without this, a query's own text is a guaranteed literal
    # substring hit against that folder, which short-circuits vault_search's keyword
    # fallback (it only engages when the ripgrep/literal leg returns zero matches) and
    # starves the RRF keyword leg inside vault_query. This mutates only this process's
    # in-memory config, not the live server, so it's safe to do unconditionally.
    config.EXCLUDED_DIRS = config.EXCLUDED_DIRS | {"retrieval-eval"}

    return config, src_root


def _import_tool(module_name: str, attr: str):
    try:
        module = importlib.import_module(f"obsidian_vault_mcp.tools.{module_name}")
        return getattr(module, attr), module
    except Exception as e:  # noqa: BLE001 - eval runner must not crash on a missing tool
        print(f"[warn] could not import {module_name}.{attr}: {e}", file=sys.stderr)
        return None, None


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH) as f:
        return yaml.safe_load(f)


def _extract_paths(raw_json: str) -> list[str]:
    """Normalise the three different result shapes into an ordered list of paths."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return []

    if isinstance(data, list):
        # vault_semantic_search: bare list of {"path": ...}
        paths = [item["path"] for item in data if "path" in item]
    elif isinstance(data, dict):
        if data.get("status") == "building":
            return []
        # vault_search / vault_query: {"results": [{"path": ...}, ...]}
        results = data.get("results", [])
        paths = [item["path"] for item in results if "path" in item]
    else:
        return []

    return [p for p in paths if not p.startswith(_SELF_REFERENTIAL_PREFIX)]


def _score_query(returned_paths: list[str], expected_paths: list[str]) -> tuple[float, float]:
    """Returns (hit_at_5, reciprocal_rank) for one query against one tool's results."""
    top_k = returned_paths[:R_AT_K]
    hit = 1.0 if any(p in expected_paths for p in top_k) else 0.0

    rr = 0.0
    for rank, path in enumerate(returned_paths, start=1):
        if path in expected_paths:
            rr = 1.0 / rank
            break

    return hit, rr


def run_tool_eval(tool_fn, call_kwargs_fn, eval_set: list[dict]) -> dict:
    """Run one tool against every question. Returns per-category + overall R@5/MRR."""
    per_category: dict[str, list[tuple[float, float]]] = {}

    for entry in eval_set:
        raw = tool_fn(entry["question"], **call_kwargs_fn())
        paths = _extract_paths(raw)
        hit, rr = _score_query(paths, entry["expected_paths"])
        per_category.setdefault(entry["category"], []).append((hit, rr))

    summary = {}
    all_scores: list[tuple[float, float]] = []
    for category, scores in per_category.items():
        hits, rrs = zip(*scores)
        summary[category] = {
            "r_at_5": round(statistics.mean(hits), 4),
            "mrr": round(statistics.mean(rrs), 4),
            "n": len(scores),
        }
        all_scores.extend(scores)

    hits, rrs = zip(*all_scores)
    summary["overall"] = {
        "r_at_5": round(statistics.mean(hits), 4),
        "mrr": round(statistics.mean(rrs), 4),
        "n": len(all_scores),
    }
    return summary


def build_tool_runners(config, src_root) -> dict:
    """Returns {tool_name: (callable, kwargs_fn)} for every tool that imports cleanly."""
    runners = {}

    vault_search, _ = _import_tool("search", "vault_search")
    if vault_search:
        runners["vault_search"] = (
            lambda q, **kw: vault_search(q, max_results=TOP_N),
            lambda: {},
        )

    vault_semantic_search, ss_module = _import_tool("semantic_search", "vault_semantic_search")
    if vault_semantic_search and ss_module is not None:
        print("Building semantic index (incremental)...", file=sys.stderr)
        ss_module.build_index()
        if ss_module._index_ready:
            runners["vault_semantic_search"] = (
                lambda q, **kw: vault_semantic_search(q, max_results=TOP_N),
                lambda: {},
            )
        else:
            print("[warn] semantic index not ready after build_index() — skipping vault_semantic_search", file=sys.stderr)

    vault_query, _ = _import_tool("query", "vault_query")
    if vault_query:
        runners["vault_query"] = (
            lambda q, **kw: vault_query(q, top_k=TOP_N),
            lambda: {},
        )
    else:
        print("[info] vault_query not deployed in this checkout — skipped", file=sys.stderr)

    return runners


def load_previous_history() -> dict | None:
    if not HISTORY_DIR.is_dir():
        return None
    files = sorted(HISTORY_DIR.glob("*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def save_history(date_str: str, results: dict) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_DIR / f"{date_str}.json", "w") as f:
        json.dump(results, f, indent=2)


def _fmt_delta(current: float, previous: float | None) -> str:
    if previous is None:
        return "baseline"
    delta = current - previous
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def render_report(date_str: str, results: dict, previous: dict | None) -> str:
    tools = list(results.keys())
    lines = [
        "---",
        "tags:",
        "  - infrastructure",
        "  - vault-mcp",
        "  - evaluation",
        "  - retrieval-eval",
        "type: reference",
        "build_id: vault-retrieval-eval",
        f"updated: '{date_str}'",
        "---",
        "",
        f"# Retrieval Eval Report — {date_str}",
        "",
        "## What this means",
        "",
        "Weekly R@5 (is the right file in the top 5 results?) and MRR (mean reciprocal rank) "
        "scores for every deployed vault retrieval tool, run against the 40-question answer key "
        "in `evals/eval-set.yaml`. Deltas are vs. the immediately preceding run. Use this to tell "
        "whether a retrieval change actually helped instead of guessing.",
        "",
        "---",
        "",
        "## Overall",
        "",
        "| Tool | R@5 | MRR | Δ R@5 | Δ MRR | n |",
        "|------|-----|-----|-------|-------|---|",
    ]

    for tool in tools:
        overall = results[tool]["overall"]
        prev_overall = (previous or {}).get(tool, {}).get("overall")
        d_r5 = _fmt_delta(overall["r_at_5"], prev_overall["r_at_5"] if prev_overall else None)
        d_mrr = _fmt_delta(overall["mrr"], prev_overall["mrr"] if prev_overall else None)
        lines.append(
            f"| {tool} | {overall['r_at_5']:.4f} | {overall['mrr']:.4f} | {d_r5} | {d_mrr} | {overall['n']} |"
        )

    lines += ["", "## By category", ""]

    categories = sorted({c for tool in tools for c in results[tool] if c != "overall"})
    for category in categories:
        lines += [f"### {category}", "", "| Tool | R@5 | MRR | Δ R@5 | Δ MRR | n |", "|------|-----|-----|-------|-------|---|"]
        for tool in tools:
            cat_scores = results[tool].get(category)
            if not cat_scores:
                continue
            prev_cat = (previous or {}).get(tool, {}).get(category)
            d_r5 = _fmt_delta(cat_scores["r_at_5"], prev_cat["r_at_5"] if prev_cat else None)
            d_mrr = _fmt_delta(cat_scores["mrr"], prev_cat["mrr"] if prev_cat else None)
            lines.append(
                f"| {tool} | {cat_scores['r_at_5']:.4f} | {cat_scores['mrr']:.4f} | {d_r5} | {d_mrr} | {cat_scores['n']} |"
            )
        lines.append("")

    if previous is None:
        lines.append("_No prior run found — this is the baseline._")

    return "\n".join(lines) + "\n"


def main() -> None:
    config, src_root = _load_live_tools()
    print(f"Tool source: {src_root}", file=sys.stderr)
    print(f"VAULT_PATH: {config.VAULT_PATH}", file=sys.stderr)

    eval_set = load_eval_set()
    assert len(eval_set) == 40, f"expected 40 eval questions, found {len(eval_set)}"

    runners = build_tool_runners(config, src_root)
    if not runners:
        print("[error] no retrieval tools importable — aborting", file=sys.stderr)
        sys.exit(1)

    results = {}
    for tool_name, (tool_fn, kwargs_fn) in runners.items():
        print(f"Scoring {tool_name}...", file=sys.stderr)
        results[tool_name] = run_tool_eval(tool_fn, kwargs_fn, eval_set)

    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    previous = load_previous_history()
    save_history(date_str, results)

    report = render_report(date_str, results, previous)
    report_path = config.VAULT_PATH / REPORT_DIR_IN_VAULT / f"{date_str}-report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"Report written to {report_path}", file=sys.stderr)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
