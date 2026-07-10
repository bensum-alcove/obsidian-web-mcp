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
from .search import _search_ripgrep, _search_python, _search_keyword_fallback

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

STALE_DAYS = 45
RRF_K = 60
_HOT_MD_MAX_BYTES = 3072
_ANSWER_CONTEXT_MAX_HOT = 3
_SUPERSEDED_STATUSES = {"superseded", "deprecated", "archived"}


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


def _rrf_fuse(keyword_paths: list[str], semantic_paths: list[str], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: score = sum of 1/(k + rank) across legs, rank is 1-indexed."""
    scores: dict[str, float] = {}
    for rank, path in enumerate(keyword_paths, start=1):
        scores[path] = scores.get(path, 0.0) + 1.0 / (k + rank)
    for rank, path in enumerate(semantic_paths, start=1):
        scores[path] = scores.get(path, 0.0) + 1.0 / (k + rank)
    return scores


def _keyword_leg(query: str, file_pattern: str, fetch_n: int) -> list[tuple[str, int]]:
    """Ranked list of (path, line_number) — first occurrence per file, in match order."""
    search_path = config.VAULT_PATH

    if shutil.which("rg"):
        matches = _search_ripgrep(query, search_path, file_pattern, fetch_n, 1)
    else:
        matches = _search_python(query, search_path, file_pattern, fetch_n, 1)

    if not matches:
        matches = _search_keyword_fallback(query, search_path, file_pattern, fetch_n, 1)

    seen: dict[str, int] = {}
    order: list[str] = []
    for m in matches:
        p = m["path"]
        if p not in seen:
            seen[p] = m["line_number"]
            order.append(p)

    return [(p, seen[p]) for p in order]


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
        fetch_n = min(300, max(50, top_k * 10))

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

            fused_score = score * _decay_factor(path, age_days) if decay else score

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
