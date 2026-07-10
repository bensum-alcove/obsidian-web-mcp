"""Management tools for the Obsidian vault MCP server."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..config import EXCLUDED_DIRS, VAULT_PATH
from ..vault import list_directory, move_path, delete_path, resolve_vault_path, _iso_timestamp

logger = logging.getLogger(__name__)


def _read_frontmatter_fields(file_path: Path, fields: list[str]) -> dict:
    """Read up to 4KB of a file and extract the requested frontmatter fields."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            head = f.read(4096)
        if not head.startswith("---"):
            return {}
        end = head.find("\n---", 3)
        if end == -1:
            return {}
        fm_text = head[3:end].strip()
        data = yaml.safe_load(fm_text) or {}
        return {k: data[k] for k in fields if k in data}
    except Exception:
        return {}


def vault_list(
    path: str = "",
    depth: int = 1,
    include_files: bool = True,
    include_dirs: bool = True,
    pattern: str | None = None,
    frontmatter_fields: list[str] | None = None,
) -> str:
    """List directory contents in the vault."""
    try:
        items = list_directory(
            path,
            depth=depth,
            include_files=include_files,
            include_dirs=include_dirs,
            pattern=pattern,
        )

        if frontmatter_fields:
            vault_root = VAULT_PATH.resolve()
            for item in items:
                if item.get("type") == "file" and item.get("name", "").endswith(".md"):
                    abs_path = vault_root / item["path"]
                    fm = _read_frontmatter_fields(abs_path, frontmatter_fields)
                    if fm:
                        item["frontmatter"] = fm

        return json.dumps({"items": items, "total": len(items)})
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except FileNotFoundError:
        return json.dumps({"error": f"Directory not found: {path}"})
    except Exception as e:
        logger.error(f"vault_list error: {e}")
        return json.dumps({"error": str(e)})


def vault_move(source: str, destination: str, create_dirs: bool = True) -> str:
    """Move a file or directory within the vault."""
    try:
        moved = move_path(source, destination, create_dirs=create_dirs)
        return json.dumps({"source": source, "destination": destination, "moved": moved})
    except ValueError as e:
        return json.dumps({"error": str(e), "source": source, "destination": destination})
    except Exception as e:
        logger.error(f"vault_move error: {e}")
        return json.dumps({"error": str(e), "source": source, "destination": destination})


def vault_delete(path: str, confirm: bool = False) -> str:
    """Delete a file by moving it to .trash/ in the vault."""
    if not confirm:
        return json.dumps({
            "error": "Set confirm=true to execute deletion. Files are moved to .trash/, not hard deleted.",
            "path": path,
        })

    try:
        deleted = delete_path(path)
        return json.dumps({"path": path, "deleted": deleted})
    except ValueError as e:
        return json.dumps({"error": str(e), "path": path})
    except Exception as e:
        logger.error(f"vault_delete error: {e}")
        return json.dumps({"error": str(e), "path": path})


def vault_batch_delete(paths: list[str], confirm: bool = False) -> str:
    """Delete multiple files by moving them to .trash/ in one call."""
    if not confirm:
        return json.dumps({
            "error": "Set confirm=true to execute deletions. Files are moved to .trash/, not hard deleted.",
            "paths": paths,
        })

    results = []
    deleted_count = 0
    failed_count = 0

    for path in paths:
        try:
            delete_path(path)
            results.append({"path": path, "deleted": True})
            deleted_count += 1
        except (ValueError, FileNotFoundError) as e:
            results.append({"path": path, "deleted": False, "error": str(e)})
            failed_count += 1
        except Exception as e:
            logger.error(f"vault_batch_delete error for {path}: {e}")
            results.append({"path": path, "deleted": False, "error": str(e)})
            failed_count += 1

    return json.dumps({"results": results, "deleted": deleted_count, "failed": failed_count})


def _os_walk(root: Path):
    """os.walk wrapper that yields (Path, list[str], list[str])."""
    import os
    for dirpath, dirnames, filenames in os.walk(root):
        yield Path(dirpath), dirnames, filenames


def vault_recent_changes(since: str, limit: int = 20) -> str:
    """Return vault .md files modified after a given ISO datetime."""
    try:
        since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        since_ts = since_dt.timestamp()
    except ValueError as e:
        return json.dumps({"error": f"Invalid ISO datetime: {e}"})

    vault_root = VAULT_PATH.resolve()
    files = []

    for dir_path, dirnames, filenames in _os_walk(vault_root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for filename in filenames:
            if not filename.endswith(".md"):
                continue
            abs_path = dir_path / filename
            try:
                stat = abs_path.stat()
                if stat.st_mtime > since_ts:
                    rel = str(abs_path.relative_to(vault_root))
                    files.append({
                        "path": rel,
                        "size": stat.st_size,
                        "modified": _iso_timestamp(stat.st_mtime),
                    })
            except OSError:
                continue

    files.sort(key=lambda f: f["modified"], reverse=True)
    total = len(files)
    files = files[:limit]

    return json.dumps({"files": files, "total": total, "since": since})


def vault_session_start(since: str | None = None) -> str:
    """Bundle session-start data in one call: stats + recent changes + manifest summary + schema pointer."""
    from datetime import timedelta
    result: dict = {}

    # Stats
    try:
        result["stats"] = json.loads(vault_stats())
    except Exception as e:
        result["stats"] = {"error": str(e)}

    # Recent changes — default 7 days ago
    try:
        if since is None:
            since_dt = datetime.now(tz=timezone.utc) - timedelta(days=7)
            since = since_dt.isoformat()
        result["recent_changes"] = json.loads(vault_recent_changes(since, limit=50))
    except Exception as e:
        result["recent_changes"] = {"error": str(e)}

    # Manifest summary (optional Build B artefact — absent gracefully)
    try:
        manifest_path = VAULT_PATH / "_manifest.json"
        if manifest_path.exists():
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            result["manifest_summary"] = {
                "generated_at": manifest_data.get("generated_at"),
                "file_count": manifest_data.get("file_count"),
                "stale_files": manifest_data.get("stale_files", []),
            }
    except Exception:
        pass

    # Schema pointer — tells Claude where vault rules live (path relative to vault root)
    result["schema_pointer"] = {
        "schema_path": "_SCHEMA.md",
        "note": "Read _SCHEMA.md for all write rules and vault conventions.",
    }

    return json.dumps(result)


def vault_stats() -> str:
    """Return vault-wide aggregate statistics."""
    vault_root = VAULT_PATH.resolve()
    all_files = []

    for dirpath, dirnames, filenames in _os_walk(vault_root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for filename in filenames:
            if not filename.endswith(".md"):
                continue
            abs_path = dirpath / filename
            try:
                stat = abs_path.stat()
                rel = str(abs_path.relative_to(vault_root))
                all_files.append({
                    "path": rel,
                    "size": stat.st_size,
                    "modified": _iso_timestamp(stat.st_mtime),
                })
            except OSError:
                continue

    total_files = len(all_files)
    total_size_kb = sum(f["size"] for f in all_files) // 1024

    largest = sorted(all_files, key=lambda f: f["size"], reverse=True)[:10]
    most_recent = sorted(all_files, key=lambda f: f["modified"], reverse=True)[:10]

    return json.dumps({
        "total_files": total_files,
        "total_size_kb": total_size_kb,
        "largest_files": largest,
        "most_recent": most_recent,
    })
