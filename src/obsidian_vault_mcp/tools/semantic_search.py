"""vault_semantic_search — fastembed + sqlite-vec semantic search over the vault.

Disabled gracefully if fastembed or sqlite-vec are not installed.
Index path: {VAULT_PATH}/.semantic-index/index.db
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Optional

try:
    from fastembed import TextEmbedding
    import sqlite_vec
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logging.warning("fastembed or sqlite-vec not available — vault_semantic_search disabled")

from .. import config

logger = logging.getLogger(__name__)

_model: Optional["TextEmbedding"] = None
_model_lock = threading.Lock()
_build_lock = threading.Lock()
_index_ready = False

# Max words per chunk before splitting at paragraph boundaries
_MAX_CHUNK_WORDS = 500

# vault-retrieval-candidate-recall-v1: schema-version bump marker used only in
# comments/tests -- the DB schema itself is unchanged (still `chunks`/`vec_chunks`),
# but a mtime+hash-keyed file is only skipped as "already indexed" if it was
# indexed by the *current* chunking logic. Chunking is a pure function of file
# content, so a stored file_hash already disambiguates old-vs-new chunking as
# long as re-chunking naturally happens on the same content -- but since the
# bug being fixed here is "the SAME content chunks differently now", any
# already-indexed vault must be force-rebuilt once after this change (see
# rebuild_all() below) rather than relying on the incremental mtime/hash skip.


_FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)---\n?", re.DOTALL)

# Frontmatter keys worth surfacing as a short natural-language prefix on the
# first chunk's embedded text -- vault-retrieval-candidate-recall-v1. Raw YAML
# frontmatter (`component_id: brain-dashboard`, `state: active`) was previously
# left inline in the first chunk's text, embedded as literal YAML syntax noise
# rather than semantic content -- diluting, not helping, the embedding. This
# reformats the handful of high-signal identity/state fields into plain prose
# ("Component: brain-dashboard. Current state: active.") prepended to the first
# chunk, and drops the frontmatter block from the embedded text entirely
# otherwise. Deliberately a small, fixed allowlist (not "dump all frontmatter")
# so non-identity metadata (revision hashes, sync_status plumbing) doesn't add
# more noise than it removes.
_FRONTMATTER_PREFIX_FIELDS = ("component_id", "type", "state", "status")


def _strip_frontmatter(content: str) -> tuple[str, dict]:
    """Returns (body_without_frontmatter, frontmatter_dict). Never raises --
    malformed/absent frontmatter just means an empty dict and unmodified content."""
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return content, {}
    try:
        import yaml
        fm = yaml.safe_load(m.group(1)) or {}
        if not isinstance(fm, dict):
            fm = {}
    except Exception:
        fm = {}
    return content[m.end():], fm


def _frontmatter_prefix_text(frontmatter: dict) -> str:
    """Short plain-English prefix from a fixed allowlist of identity/state
    frontmatter fields -- see _FRONTMATTER_PREFIX_FIELDS. Empty string if none present."""
    parts = []
    for field in _FRONTMATTER_PREFIX_FIELDS:
        value = frontmatter.get(field)
        if value:
            label = field.replace("_", " ").capitalize()
            parts.append(f"{label}: {value}.")
    return " ".join(parts)


def _get_model() -> "TextEmbedding":
    global _model
    with _model_lock:
        if _model is None:
            _model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def _get_index_path() -> Path:
    index_dir = config.VAULT_PATH / ".semantic-index"
    index_dir.mkdir(exist_ok=True)
    return index_dir / "index.db"


def _open_db() -> sqlite3.Connection:
    db = sqlite3.connect(str(_get_index_path()))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("PRAGMA journal_mode=WAL")
    return db


def _ensure_schema(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            mtime REAL NOT NULL,
            file_hash TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            section_heading TEXT,
            content TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[384] distance_metric=cosine
        );
    """)
    db.commit()


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _serialize(emb) -> bytes:
    """Serialize a float iterable to little-endian float32 bytes for sqlite-vec."""
    floats = [float(x) for x in emb]
    return struct.pack(f"{len(floats)}f", *floats)


# --- Table-row sub-chunking (vault-retrieval-candidate-recall-v1) ----------
#
# Diagnosis (see this build's output doc): dense reference pages like
# SYSTEM-FACTS.md pack many unrelated atomic facts into ONE `##`-delimited
# section as a markdown table (`| Fact | Value |` rows) -- e.g. one single
# "Edge Trading System" chunk contains the SPI dashboard URL, IBKR client IDs,
# AND the trade-score-gate config knob, all averaged into one embedding. A
# query about just the trade-score gate has to compete, in that chunk's
# embedding, against 7 other unrelated facts' worth of signal -- confirmed by
# rank-tracing the frozen v3 paraphrase questions: the correct document was
# completely absent from the top-150 semantic candidates for two such
# questions. This does NOT change any non-table content's chunking at all --
# a section is only split into extra sub-chunks when it contains a markdown
# table where a per-row split is unambiguous (a leading `|` line followed by a
# separator row of dashes/pipes), so no impact on regular prose sections.

_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_MIN_TABLE_ROWS_TO_SPLIT = 4  # below this, per-row chunks add DB/latency cost for no measured benefit


def _split_table_rows(text: str) -> Optional[list[str]]:
    """If `text` is (mostly) a single markdown table, returns one string per
    data row (each prefixed with its column headers as plain text, e.g.
    "Fact: 502 from vault URL. Value: MCP supervisord process is down, not the
    tunnel."), plus any leading/trailing non-table prose as separate entries.
    Returns None if this isn't a table-shaped section (leaves it untouched --
    the caller falls back to the existing whole-section chunking).
    """
    lines = text.splitlines()
    table_start = None
    for i, line in enumerate(lines):
        if _TABLE_ROW_RE.match(line) and i + 1 < len(lines) and _TABLE_SEPARATOR_RE.match(lines[i + 1]):
            table_start = i
            break
    if table_start is None:
        return None

    header_cells = [c.strip() for c in lines[table_start].strip().strip("|").split("|")]
    table_end = table_start + 2
    data_rows = []
    while table_end < len(lines) and _TABLE_ROW_RE.match(lines[table_end]):
        data_rows.append(lines[table_end])
        table_end += 1

    if len(data_rows) < _MIN_TABLE_ROWS_TO_SPLIT:
        return None

    entries: list[str] = []
    pre_table = "\n".join(lines[:table_start]).strip()
    if pre_table:
        entries.append(pre_table)

    for row in data_rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        pieces = []
        for header, cell in zip(header_cells, cells):
            if header and cell:
                pieces.append(f"{header}: {cell}.")
        if pieces:
            entries.append(" ".join(pieces))
        elif row.strip():
            entries.append(row.strip())

    post_table = "\n".join(lines[table_end:]).strip()
    if post_table:
        entries.append(post_table)

    return entries if entries else None


def _chunk_text(content: str, frontmatter: Optional[dict] = None) -> list[dict]:
    """Split by ## headings; split oversized sections at paragraph boundaries;
    split dense fact tables into per-row sub-chunks (see _split_table_rows).

    `content` is expected to already have frontmatter stripped by the caller
    (build_index/reindex_paths do this via _strip_frontmatter) -- accepting the
    already-stripped body here, with `frontmatter` passed separately, keeps this
    function a pure function of (body, frontmatter) rather than re-parsing YAML
    per call.
    """
    if frontmatter is None:
        frontmatter = {}

    sections: list[tuple[Optional[str], list[str]]] = []
    current_heading: Optional[str] = None
    current_lines: list[str] = []

    for line in content.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_lines))

    chunks: list[dict] = []
    for heading, lines in sections:
        text = "\n".join(lines).strip()
        if not text:
            continue

        table_entries = _split_table_rows(text) if config.VAULT_SEMANTIC_TABLE_ROW_CHUNKING else None
        if table_entries is not None:
            for entry in table_entries:
                chunks.append({"heading": heading, "content": entry})
            continue

        if len(text.split()) <= _MAX_CHUNK_WORDS:
            chunks.append({"heading": heading, "content": text})
        else:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            current_paras: list[str] = []
            current_count = 0
            for para in paragraphs:
                para_words = len(para.split())
                if current_count + para_words > _MAX_CHUNK_WORDS and current_paras:
                    chunks.append({"heading": heading, "content": "\n\n".join(current_paras)})
                    current_paras = [para]
                    current_count = para_words
                else:
                    current_paras.append(para)
                    current_count += para_words
            if current_paras:
                chunks.append({"heading": heading, "content": "\n\n".join(current_paras)})

    if chunks and config.VAULT_SEMANTIC_FRONTMATTER_PREFIX:
        prefix = _frontmatter_prefix_text(frontmatter)
        if prefix:
            chunks[0] = {
                "heading": chunks[0]["heading"],
                "content": f"{prefix} {chunks[0]['content']}".strip(),
            }

    return chunks


def _prepare_content_for_chunking(raw_content: str) -> tuple[str, dict]:
    """Frontmatter stripped from the embedded text (vault-retrieval-candidate-
    recall-v1) -- raw YAML syntax was previously embedded verbatim as part of
    the first chunk, contributing tokenizer/embedding noise rather than
    semantic signal. Opt-out via VAULT_SEMANTIC_STRIP_FRONTMATTER; disabled
    reproduces the previous byte-identical behaviour (frontmatter left inline)."""
    if not config.VAULT_SEMANTIC_STRIP_FRONTMATTER:
        return raw_content, {}
    return _strip_frontmatter(raw_content)


def build_index() -> None:
    """Build/update the semantic index. Blocking — must be called via asyncio.to_thread()."""
    global _index_ready

    if not SEMANTIC_AVAILABLE:
        return

    with _build_lock:
        logger.info("Semantic index build starting...")
        try:
            db = _open_db()
            _ensure_schema(db)
            model = _get_model()
            vault_path = config.VAULT_PATH

            # Current index state: file_path -> (mtime, file_hash)
            indexed = {
                row[0]: (row[1], row[2])
                for row in db.execute(
                    "SELECT DISTINCT file_path, mtime, file_hash FROM chunks"
                ).fetchall()
            }

            # Discover vault .md files, excluding all EXCLUDED_DIRS components
            vault_files: dict[str, Path] = {}
            for md_file in vault_path.rglob("*.md"):
                rel = str(md_file.relative_to(vault_path))
                if any(part in config.RETRIEVAL_EXCLUDED_DIRS for part in Path(rel).parts):
                    continue
                vault_files[rel] = md_file

            # Remove stale entries for deleted files
            deleted = set(indexed) - set(vault_files)
            for path in deleted:
                db.execute(
                    "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?)",
                    (path,),
                )
                db.execute("DELETE FROM chunks WHERE file_path = ?", (path,))
            if deleted:
                db.commit()
                logger.info(f"Removed {len(deleted)} deleted files from index")

            # Index new/changed files
            updated = 0
            for rel, md_file in vault_files.items():
                try:
                    mtime = md_file.stat().st_mtime
                    raw_content = md_file.read_text(encoding="utf-8", errors="replace")
                    fhash = _file_hash(raw_content)

                    if rel in indexed and indexed[rel][0] == mtime and indexed[rel][1] == fhash:
                        continue

                    db.execute(
                        "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?)",
                        (rel,),
                    )
                    db.execute("DELETE FROM chunks WHERE file_path = ?", (rel,))

                    body, frontmatter = _prepare_content_for_chunking(raw_content)
                    file_chunks = _chunk_text(body, frontmatter)
                    if not file_chunks:
                        continue

                    embeddings = list(model.embed([c["content"] for c in file_chunks]))

                    for i, (chunk, emb) in enumerate(zip(file_chunks, embeddings)):
                        cur = db.execute(
                            "INSERT INTO chunks (file_path, mtime, file_hash, chunk_index, section_heading, content)"
                            " VALUES (?, ?, ?, ?, ?, ?)",
                            (rel, mtime, fhash, i, chunk["heading"], chunk["content"]),
                        )
                        db.execute(
                            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                            (cur.lastrowid, _serialize(emb)),
                        )

                    db.commit()
                    updated += 1

                except Exception as e:
                    logger.warning(f"Failed to index {rel}: {e}")

            db.close()
            _index_ready = True
            logger.info(
                f"Semantic index build complete: {updated} files updated, {len(vault_files)} total"
            )

        except Exception as e:
            logger.error(f"Semantic index build failed: {e}", exc_info=True)


def vault_semantic_search(
    query: str,
    max_results: int = 5,
    path_prefix: Optional[str] = None,
) -> str:
    """Synchronous search implementation — call via asyncio.to_thread() from async context."""
    if not _index_ready:
        return json.dumps({
            "status": "building",
            "message": (
                "Semantic index is building. Try again in a moment, "
                "or use vault_search for keyword search."
            ),
        })

    try:
        model = _get_model()
        query_emb = next(model.embed([query]))

        db = _open_db()
        # vault-retrieval-candidate-recall-v1: floor raised via config (see
        # VAULT_SEMANTIC_FETCH_MULTIPLIER/_MIN) -- a bare max_results*5 fetch depth
        # (e.g. 25 for the default max_results=5) put several correct paraphrase-
        # category documents outside the KNN window entirely; this floor is a
        # superset (always >= the old multiplier-only value), never a reduction.
        fetch_n = max(max_results * config.VAULT_SEMANTIC_FETCH_MULTIPLIER, config.VAULT_SEMANTIC_FETCH_MIN)

        knn_rows = db.execute(
            "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?",
            (_serialize(query_emb), fetch_n),
        ).fetchall()

        results: list[tuple[str, Optional[str], str, float]] = []
        for chunk_id, distance in knn_rows:
            row = db.execute(
                "SELECT file_path, section_heading, content FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
            if row:
                results.append((row[0], row[1], row[2], float(distance)))

        db.close()

        if path_prefix:
            results = [r for r in results if r[0].startswith(path_prefix)]

        output = []
        for file_path, section_heading, content, distance in results[:max_results]:
            snippet = content[:300].replace("\n", " ").strip()
            if len(content) > 300:
                snippet += "…"
            output.append({
                "path": file_path,
                "score": round(1.0 - distance / 2.0, 4),
                "snippet": snippet,
                "section": section_heading or "",
            })

        return json.dumps(output)

    except Exception as e:
        logger.error(f"Semantic search failed: {e}", exc_info=True)
        return json.dumps({"error": str(e)})
