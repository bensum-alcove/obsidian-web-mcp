"""vault_query and vault_answer_context — fused hybrid retrieval for the Obsidian vault MCP server.

Collapses the semantic-vs-keyword tool choice into one call: reuses the existing
ripgrep search leg and the existing semantic embedding index, fuses both with
Reciprocal Rank Fusion, applies optional temporal decay, and flags stale/archived
content. vault_answer_context wraps vault_query with a hot.md bundle for a
one-call pre-flight read.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import frontmatter as fm_lib

from .. import config
from ..utils import sanitize_for_json, SafeJSONEncoder
from .search import (
    _search_ripgrep,
    _search_python,
    _search_keyword_fallback,
    _search_by_tokens,
    _tokenize_query,
)

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

STALE_DAYS = 45
_HOT_MD_MAX_BYTES = 3072
_ANSWER_CONTEXT_MAX_HOT = 3
_SUPERSEDED_STATUSES = {"superseded", "deprecated", "archived"}

# vault_query keyword leg tuning. A query at or under this many extracted content
# tokens is short enough that a raw-sentence ripgrep pattern is effectively already
# an exact-phrase search (the vault_search-style "2-4 nouns" case) -- leave that path
# untouched. Above the threshold, the raw sentence stops being a reliable literal/regex
# pattern, so the per-token merge below takes over when the raw-sentence leg is weak.
_SHORT_QUERY_TOKEN_THRESHOLD = 4

# A raw-sentence keyword-leg result counts as "weak" -- and triggers the per-token
# merge -- if it has fewer than this many distinct-file matches, or if every match it
# does have only overlaps one query token. Fewer than 3 files is too thin a candidate
# set to trust over a full per-token search; a match containing only one query token
# is exactly the "incidental regex hit on a full sentence" failure mode from the
# 2026-08-08 diagnosis (BS_BRAIN_API_KEY / 502 questions matched zero or one token's
# worth of the sentence and got crowded out by topically-adjacent files instead).
_WEAK_MATCH_MIN_COUNT = 3


def _dedupe_first_occurrence(matches: list[dict]) -> list[tuple[str, int]]:
    """First occurrence per file, path order preserved from the input match order."""
    seen: dict[str, int] = {}
    order: list[str] = []
    for m in matches:
        p = m["path"]
        if p not in seen:
            seen[p] = m["line_number"]
            order.append(p)
    return [(p, seen[p]) for p in order]


def _run_primary_keyword_search(query: str, search_path: Path, file_pattern: str, fetch_n: int) -> list[dict]:
    """The raw-sentence keyword search: ripgrep if available, else the Python fallback."""
    if shutil.which("rg"):
        return _search_ripgrep(query, search_path, file_pattern, fetch_n, 1)
    return _search_python(query, search_path, file_pattern, fetch_n, 1)


def _count_matched_tokens(text: str, tokens: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for t in tokens if t.lower() in text_lower)


def _is_weak_keyword_result(matches: list[dict], tokens: list[str]) -> bool:
    """See _WEAK_MATCH_MIN_COUNT for the threshold rationale."""
    if len(matches) < _WEAK_MATCH_MIN_COUNT:
        return True
    return not any(_count_matched_tokens(m.get("match_context", ""), tokens) > 1 for m in matches)


def _is_archived(path: str) -> bool:
    """True if the path lives under an _Archive/ or .trash/ directory at any level."""
    parts = Path(path).parts
    return "_Archive" in parts or ".trash" in parts


def _decay_factor(path: str, age_days: float) -> float:
    """Exponential decay multiplier; half-life picked by the longest matching path-prefix key."""
    half_life = config.VAULT_QUERY_DEFAULT_HALF_LIFE_DAYS
    best_match_len = -1
    for prefix, hl in config.VAULT_QUERY_HALF_LIFE_OVERRIDES.items():
        if prefix in path and len(prefix) > best_match_len:
            half_life = hl
            best_match_len = len(prefix)
    if half_life <= 0:
        return 1.0
    return math.exp(-age_days * math.log(2) / half_life)


def _rrf_fuse(keyword_paths: list[str], semantic_paths: list[str], k: float | None = None) -> dict[str, float]:
    """Reciprocal Rank Fusion: score = sum of 1/(k + rank) across legs, rank is 1-indexed."""
    if k is None:
        k = config.VAULT_QUERY_RRF_K
    scores: dict[str, float] = {}
    for rank, path in enumerate(keyword_paths, start=1):
        scores[path] = scores.get(path, 0.0) + 1.0 / (k + rank)
    for rank, path in enumerate(semantic_paths, start=1):
        scores[path] = scores.get(path, 0.0) + 1.0 / (k + rank)
    return scores


def _frontmatter_type(full_path: Path) -> str:
    try:
        content = full_path.read_text(encoding="utf-8", errors="replace")
        post = fm_lib.loads(content)
        return str((post.metadata or {}).get("type", "")).lower()
    except Exception:
        return ""


_CANONICAL_STATE_DIR = "Canonical State/records/"


def _canonical_boost_factor(path: str) -> float:
    """See VAULT_QUERY_CANONICAL_BOOST in config.py for the rationale. Cheap path-
    substring pre-check before the frontmatter parse -- canonical-state records live
    under a known directory convention (verified against every `type: canonical-state`
    file in this build's diagnosis), so this skips a disk read + YAML parse for the
    large majority of fused candidates that obviously aren't canonical records. Falls
    through to nothing (no boost, not a correctness issue) for a hypothetical
    canonical-state file living outside that directory -- the frontmatter check
    remains the authority, this is purely a latency shortcut."""
    boost = config.VAULT_QUERY_CANONICAL_BOOST
    if boost == 1.0 or _CANONICAL_STATE_DIR not in path:
        return 1.0
    full_path = config.VAULT_PATH / path
    if _frontmatter_type(full_path) == "canonical-state":
        return boost
    return 1.0


def _keyword_leg(query: str, file_pattern: str, fetch_n: int) -> list[tuple[str, int]]:
    """Ranked list of (path, line_number) — first occurrence per file, in match order."""
    search_path = config.VAULT_PATH
    matches = _run_primary_keyword_search(query, search_path, file_pattern, fetch_n)

    if not config.VAULT_QUERY_KEYWORD_TOKENIZE:
        # Kill switch off: byte-identical to pre-tokenization behaviour.
        if not matches:
            matches = _search_keyword_fallback(query, search_path, file_pattern, fetch_n, 1)
        return _dedupe_first_occurrence(matches)

    tokens = _tokenize_query(query)

    if len(tokens) <= _SHORT_QUERY_TOKEN_THRESHOLD:
        # Short queries: identical behaviour to today (and to the kill-switch-off path).
        if not matches:
            matches = _search_keyword_fallback(query, search_path, file_pattern, fetch_n, 1)
        return _dedupe_first_occurrence(matches)

    # Long queries: a raw full-sentence match is trusted only if it's strong.
    # Otherwise, search per-token and merge, ranking by distinct-token overlap.
    # require_all=True: matches _search_keyword_fallback's long-standing AND-
    # preferred/OR-fallback behaviour. require_all=False (pure count ranking,
    # no AND gate) was tried and measured against the eval: it fixed some
    # partial-match cases but let large, topically broad files accumulate a
    # high raw token count just from size and crowd out narrower, more
    # relevant files even more often than the AND gate does -- net worse on
    # the eval. require_all=True is the better-measured tradeoff of the two.
    #
    # allow_partial=True (vault-retrieval-candidate-recall-v1): on top of the
    # AND-preferred/OR-fallback behaviour above, files satisfying most-but-not-
    # all tokens are appended as lower-ranked candidates rather than being
    # invisible to the keyword leg entirely -- see _search_by_tokens' docstring
    # for the diagnosed candidate-absence failure this closes (a small, correct
    # document missing one query token loses outright to a huge, topically
    # broad file that happens to contain every token somewhere in its size).
    if _is_weak_keyword_result(matches, tokens):
        matches = _search_by_tokens(
            [t.lower() for t in tokens], search_path, file_pattern, fetch_n, 1,
            require_all=True,
            allow_partial=config.VAULT_QUERY_ALLOW_PARTIAL_KEYWORD_MATCH,
        )

    return _dedupe_first_occurrence(matches)


def _semantic_leg(query: str, fetch_n: int) -> list[tuple[str, str | None, str, float]]:
    """Ranked list of (path, heading, content, distance) — best chunk per file, ascending distance."""
    from . import semantic_search as ss

    if not ss.SEMANTIC_AVAILABLE or not ss._index_ready:
        return []

    db = None
    try:
        model = ss._get_model()
        query_emb = next(model.embed([query]))

        db = ss._open_db()
        knn_rows = db.execute(
            "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?",
            (ss._serialize(query_emb), fetch_n),
        ).fetchall()

        best_per_file: dict[str, tuple[str | None, str, float]] = {}
        order: list[str] = []
        for chunk_id, distance in knn_rows:
            row = db.execute(
                "SELECT file_path, section_heading, content FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
            if not row:
                continue
            file_path, heading, content = row
            distance = float(distance)
            if file_path not in best_per_file:
                order.append(file_path)
                best_per_file[file_path] = (heading, content, distance)
            elif distance < best_per_file[file_path][2]:
                best_per_file[file_path] = (heading, content, distance)

        ranked = sorted(order, key=lambda p: best_per_file[p][2])
        return [(p, best_per_file[p][0], best_per_file[p][1], best_per_file[p][2]) for p in ranked]

    except Exception as e:
        logger.warning("vault_query semantic leg failed: %s", e)
        return []
    finally:
        if db is not None:
            db.close()


def _nearest_heading(full_path: Path, line_number: int) -> str | None:
    """Scan backward from line_number for the nearest markdown heading."""
    try:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if not lines:
        return None
    idx = min(max(line_number - 1, 0), len(lines) - 1)
    for i in range(idx, -1, -1):
        m = _HEADING_RE.match(lines[i].strip())
        if m:
            return m.group(2).strip()
    return None


def _line_context(full_path: Path, line_number: int, context_lines: int = 3) -> str | None:
    """Fallback chunk text for keyword-only matches with no semantic content available."""
    try:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if not lines:
        return None
    idx = min(max(line_number - 1, 0), len(lines) - 1)
    start = max(0, idx - context_lines)
    end = min(len(lines), idx + context_lines + 1)
    return "\n".join(lines[start:end])


def vault_query(
    query: str,
    top_k: int = 8,
    path_prefix: str | None = None,
    include_archive: bool = False,
    decay: bool = True,
) -> str:
    """Fused hybrid search: BM25-ish ripgrep leg + semantic leg, merged with RRF,
    temporal decay, archive exclusion, staleness flags, and expand handles."""
    try:
        # vault-retrieval-candidate-recall-v1: floor raised (see config.py's
        # VAULT_SEMANTIC_FETCH_MIN) -- diagnosis found correct paraphrase-category
        # documents at candidate rank 16-45, past the previous ~50-150 ceiling for
        # small top_k callers on some query shapes. Deeper candidate pool costs
        # latency (more DB rows + per-candidate stat() calls), not correctness.
        fetch_n = min(300, max(config.VAULT_SEMANTIC_FETCH_MIN, top_k * 10))

        keyword_hits = _keyword_leg(query, "*.md", fetch_n)
        keyword_paths = [p for p, _ in keyword_hits]
        keyword_line_by_path = dict(keyword_hits)

        semantic_hits = _semantic_leg(query, fetch_n)
        semantic_paths = [p for p, *_ in semantic_hits]
        semantic_by_path = {p: (h, c, d) for p, h, c, d in semantic_hits}

        fused = _rrf_fuse(keyword_paths, semantic_paths)

        if path_prefix:
            fused = {p: s for p, s in fused.items() if p.startswith(path_prefix)}

        if not include_archive:
            fused = {p: s for p, s in fused.items() if not _is_archived(p)}

        now = datetime.now(tz=timezone.utc)
        results = []

        for path, score in fused.items():
            full_path = config.VAULT_PATH / path
            try:
                mtime = full_path.stat().st_mtime
            except OSError:
                continue

            modified_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
            age_days = (now - modified_dt).total_seconds() / 86400.0

            boosted_score = score * _canonical_boost_factor(path)
            fused_score = boosted_score * _decay_factor(path, age_days) if decay else boosted_score

            heading = None
            chunk = None
            if path in semantic_by_path:
                heading, content, _ = semantic_by_path[path]
                chunk = content[:400].strip()
                if len(content) > 400:
                    chunk += "…"

            if path in keyword_line_by_path:
                line_number = keyword_line_by_path[path]
                if heading is None:
                    heading = _nearest_heading(full_path, line_number)
                if chunk is None:
                    chunk = _line_context(full_path, line_number)

            results.append({
                "path": path,
                "heading": heading,
                "chunk": chunk,
                "score": round(fused_score, 6),
                "updated": modified_dt.isoformat(),
                "stale": age_days > STALE_DAYS,
                "expand": {"path": path, "heading": heading} if heading else None,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        results = results[:top_k]

        return json.dumps(sanitize_for_json({
            "query": query,
            "results": results,
            "total_candidates": len(fused),
        }), cls=SafeJSONEncoder)

    except Exception as e:
        logger.error("vault_query error: %s", e)
        return json.dumps({"error": str(e)})


def _top_level_folder(path: str) -> str:
    parts = Path(path).parts
    return parts[0] if parts else ""


def _find_hot_md_files() -> list[Path]:
    return list(config.VAULT_PATH.rglob("hot.md"))


def _frontmatter_status(full_path: Path) -> str:
    try:
        content = full_path.read_text(encoding="utf-8", errors="replace")
        post = fm_lib.loads(content)
        status = (post.metadata or {}).get("status", "")
        return str(status).lower()
    except Exception:
        return ""


def vault_answer_context(question: str, top_k: int = 6) -> str:
    """One-call pre-flight bundle: vault_query(question) + up to 3 hot.md files + staleness warnings."""
    try:
        query_result = json.loads(vault_query(question, top_k=top_k))
        if "error" in query_result:
            return json.dumps(sanitize_for_json(query_result), cls=SafeJSONEncoder)

        results = query_result.get("results", [])
        top_folders = {_top_level_folder(r["path"]) for r in results if r.get("path")}

        hot_candidates = []
        for hot_path in _find_hot_md_files():
            try:
                size = hot_path.stat().st_size
            except OSError:
                continue
            if size > _HOT_MD_MAX_BYTES:
                continue
            rel = str(hot_path.relative_to(config.VAULT_PATH))
            shares_folder = _top_level_folder(rel) in top_folders
            hot_candidates.append((shares_folder, rel, hot_path))

        # Prefer hot.md files sharing a top-level folder with the top results.
        hot_candidates.sort(key=lambda x: not x[0])
        selected = hot_candidates[:_ANSWER_CONTEXT_MAX_HOT]

        hot_files = []
        for _, rel, hot_path in selected:
            try:
                content = hot_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                content = None
                logger.warning("vault_answer_context: failed reading %s: %s", rel, e)
            hot_files.append({"path": rel, "content": content})

        warnings = []
        for r in results:
            if r.get("stale"):
                warnings.append({
                    "path": r["path"],
                    "reason": "stale",
                    "detail": f"Unmodified since {r.get('updated')}",
                })
            status = _frontmatter_status(config.VAULT_PATH / r["path"])
            if status in _SUPERSEDED_STATUSES:
                warnings.append({
                    "path": r["path"],
                    "reason": "superseded",
                    "detail": f"frontmatter status: {status}",
                })

        return json.dumps(sanitize_for_json({
            "question": question,
            "results": results,
            "hot": hot_files,
            "warnings": warnings,
        }), cls=SafeJSONEncoder)

    except Exception as e:
        logger.error("vault_answer_context error: %s", e)
        return json.dumps({"error": str(e)})
