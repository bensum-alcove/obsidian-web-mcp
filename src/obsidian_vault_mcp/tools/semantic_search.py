"""vault_semantic_search — fastembed + sqlite-vec semantic search over the vault.

Disabled gracefully if fastembed or sqlite-vec are not installed.
Index path: {VAULT_PATH}/.semantic-index/index.db
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
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
from ..vault import resolve_vault_path

logger = logging.getLogger(__name__)

_model: Optional["TextEmbedding"] = None
_model_lock = threading.Lock()
_build_lock = threading.Lock()
_index_ready = False

# Max words per chunk before splitting at paragraph boundaries
_MAX_CHUNK_WORDS = 500

# Debounce state for event-triggered reindex after writes
_DEBOUNCE_SECONDS = 5.0
_pending_paths: set[str] = set()
_pending_lock = threading.Lock()
_debounce_timer: Optional[threading.Timer] = None


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


def _chunk_text(content: str) -> list[dict]:
    """Split by ## headings; split oversized sections at paragraph boundaries."""
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
    return chunks


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
                if any(part in config.EXCLUDED_DIRS for part in Path(rel).parts):
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
                    content = md_file.read_text(encoding="utf-8", errors="replace")
                    fhash = _file_hash(content)

                    if rel in indexed and indexed[rel][0] == mtime and indexed[rel][1] == fhash:
                        continue

                    db.execute(
                        "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?)",
                        (rel,),
                    )
                    db.execute("DELETE FROM chunks WHERE file_path = ?", (rel,))

                    file_chunks = _chunk_text(content)
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


def reindex_paths(paths: list[str]) -> None:
    """Incrementally re-embed specific files by path. Blocking — uses _build_lock."""
    if not SEMANTIC_AVAILABLE or not paths:
        return
    with _build_lock:
        db = None
        try:
            db = _open_db()
            _ensure_schema(db)
            model = _get_model()
            vault_path = config.VAULT_PATH
            updated = 0
            for rel in paths:
                md_file = vault_path / rel
                try:
                    if not md_file.exists():
                        db.execute(
                            "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?)",
                            (rel,),
                        )
                        db.execute("DELETE FROM chunks WHERE file_path = ?", (rel,))
                        db.commit()
                        continue
                    mtime = md_file.stat().st_mtime
                    content = md_file.read_text(encoding="utf-8", errors="replace")
                    fhash = _file_hash(content)
                    row = db.execute(
                        "SELECT mtime, file_hash FROM chunks WHERE file_path = ? LIMIT 1", (rel,)
                    ).fetchone()
                    if row and row[0] == mtime and row[1] == fhash:
                        continue
                    db.execute(
                        "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?)",
                        (rel,),
                    )
                    db.execute("DELETE FROM chunks WHERE file_path = ?", (rel,))
                    file_chunks = _chunk_text(content)
                    if not file_chunks:
                        db.commit()
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
                    db.rollback()
                    logger.warning(f"Event reindex failed for {rel}: {e}")
            if updated:
                logger.info(f"Event reindex complete: {updated} file(s) updated")
        except Exception as e:
            logger.error(f"Event reindex batch failed: {e}", exc_info=True)
        finally:
            if db is not None:
                db.close()


def _debounce_fire() -> None:
    """Timer callback — drain pending paths and reindex them."""
    global _debounce_timer
    with _pending_lock:
        paths = list(_pending_paths)
        _pending_paths.clear()
        _debounce_timer = None
    if paths:
        try:
            reindex_paths(paths)
        except Exception as e:
            logger.error(f"Debounced reindex failed: {e}")


def schedule_reindex(path: str) -> None:
    """Schedule a debounced incremental reindex for a written file.
    Safe to call from any thread. No-op if SEMANTIC_AVAILABLE is False."""
    global _debounce_timer
    if not SEMANTIC_AVAILABLE or not path:
        return
    with _pending_lock:
        _pending_paths.add(path)
        if _debounce_timer is not None:
            _debounce_timer.cancel()
        t = threading.Timer(_DEBOUNCE_SECONDS, _debounce_fire)
        t.daemon = True
        t.start()
        _debounce_timer = t


async def periodic_reindex(interval_hours: float = 0.5) -> None:
    """Re-run incremental index build on a timer."""
    while True:
        await asyncio.sleep(interval_hours * 3600)
        try:
            logger.info(f"Periodic re-index starting for {config.VAULT_PATH}")
            await asyncio.to_thread(build_index)
            logger.info(f"Periodic re-index complete for {config.VAULT_PATH}")
        except Exception as e:
            logger.error(f"Periodic re-index failed: {e}")


async def startup_then_periodic() -> None:
    """Run initial index build then schedule periodic re-index."""
    await asyncio.to_thread(build_index)
    asyncio.create_task(periodic_reindex())


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
        fetch_n = max_results * 5

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


def vault_read_smart(
    path: str,
    query: str,
    max_sections: int = 3,
) -> str:
    """Section-level RAG over a single file. Returns top max_sections chunks by semantic similarity.
    Blocking — call via asyncio.to_thread().
    Note: read_policy: section-only is intentionally not enforced — this tool IS the section
    selector, returning only relevant sections rather than the full content."""
    try:
        try:
            md_file = resolve_vault_path(path)
        except ValueError as e:
            return json.dumps({"error": str(e), "path": path})
        if not md_file.exists():
            return json.dumps({"error": f"File not found: {path}"})

        content = md_file.read_text(encoding="utf-8", errors="replace")

        # Small file: return whole thing
        if len(content) < 8192:
            return json.dumps({"path": path, "mode": "full", "content": content})

        chunks = _chunk_text(content)
        if not chunks:
            return json.dumps({"path": path, "mode": "full", "content": content})

        try:
            model = _get_model()
            all_texts = [query] + [c["content"] for c in chunks]
            all_embs = list(model.embed(all_texts))
            query_emb = all_embs[0]
            chunk_embs = all_embs[1:]

            try:
                import numpy as np
                q = np.array(query_emb, dtype=float)
                C = np.array(chunk_embs, dtype=float)
                q_norm = float(np.linalg.norm(q))
                c_norms = np.linalg.norm(C, axis=1)
                scores = ((C @ q) / (c_norms * q_norm + 1e-10)).tolist()
            except Exception:
                def _dot(a, b) -> float:
                    return sum(float(x) * float(y) for x, y in zip(a, b))
                def _norm(v) -> float:
                    return sum(float(x) * float(x) for x in v) ** 0.5
                q_norm = _norm(query_emb)
                scores = [
                    _dot(ce, query_emb) / (_norm(ce) * q_norm + 1e-10)
                    for ce in chunk_embs
                ]

            ranked = sorted(
                zip(scores, chunks),
                key=lambda x: x[0],
                reverse=True,
            )[:max_sections]

            return json.dumps({
                "path": path,
                "mode": "smart",
                "query": query,
                "sections": [
                    {
                        "heading": c["heading"] or "",
                        "score": round(float(s), 4),
                        "content": c["content"],
                    }
                    for s, c in ranked
                ],
            })

        except Exception as e:
            logger.warning(f"vault_read_smart embedding failed for {path}: {e}")
            headings = [c["heading"] for c in chunks if c["heading"]]
            return json.dumps({
                "path": path,
                "mode": "fallback",
                "note": "Embedding failed — available sections listed. Use vault_read_section to read one.",
                "sections": headings,
            })

    except Exception as e:
        logger.error(f"vault_read_smart error for {path}: {e}", exc_info=True)
        return json.dumps({"error": str(e), "path": path})
