"""vault_semantic_search — fastembed + sqlite-vec semantic search over the vault.

Disabled gracefully if fastembed or sqlite-vec are not installed.
Index path: {VAULT_PATH}/.semantic-index/index.db
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)

_model: Optional["TextEmbedding"] = None
_model_lock = threading.Lock()
_build_lock = threading.Lock()
_index_ready = False

# Max words per chunk before splitting at paragraph boundaries
_MAX_CHUNK_WORDS = 500


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
