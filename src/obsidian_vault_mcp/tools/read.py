"""Read tools for the Obsidian vault MCP server."""

import json
import logging
import re

import frontmatter

from ..vault import resolve_vault_path, read_file
from ..utils import sanitize_for_json, SafeJSONEncoder

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_MD_INLINE_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)|`([^`]+)`|\*\*([^*]+)\*\*|\*([^*]+)\*|__([^_]+)__|_([^_]+)_")


def _strip_inline_md(text: str) -> str:
    """Strip common inline markdown, returning plain text for heading comparison."""
    def _replace(m):
        for g in m.groups():
            if g is not None:
                return g
        return ""
    return _MD_INLINE_RE.sub(_replace, text).strip()


def vault_read_section(path: str, section: str) -> str:
    """Read a single markdown section by heading name."""
    try:
        content, metadata = read_file(path)

        fm_data = None
        try:
            post = frontmatter.loads(content)
            if post.metadata:
                fm_data = post.metadata
        except Exception:
            pass

        lines = content.splitlines(keepends=True)

        # Collect all headings: (line_index, level, plain_text)
        headings = []
        for i, line in enumerate(lines):
            m = _HEADING_RE.match(line.rstrip("\r\n"))
            if m:
                level = len(m.group(1))
                plain = _strip_inline_md(m.group(2))
                headings.append((i, level, plain))

        if not headings:
            return json.dumps({"error": "No sections found in file", "path": path})

        # Find first heading matching section (case-insensitive)
        target_section = section.strip().lstrip("#").strip()
        target_idx = None
        target_level = None
        for (line_i, level, plain) in headings:
            if plain.lower() == target_section.lower():
                target_idx = line_i
                target_level = level
                break

        if target_idx is None:
            available = [plain for (_, _, plain) in headings[:20]]
            return json.dumps(sanitize_for_json({
                "error": f"Section '{section}' not found",
                "path": path,
                "available_sections": available,
            }), cls=SafeJSONEncoder)

        # Find end of section: next heading of equal or higher level (same or fewer #)
        end_line = len(lines)
        for (line_i, level, plain) in headings:
            if line_i > target_idx and level <= target_level:
                end_line = line_i
                break

        section_content = "".join(lines[target_idx:end_line])

        return json.dumps(sanitize_for_json({
            "path": path,
            "section": section,
            "content": section_content,
            "metadata": metadata,
            "frontmatter": fm_data,
        }), cls=SafeJSONEncoder)
    except ValueError as e:
        return json.dumps({"error": str(e), "path": path})
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {path}", "path": path})
    except Exception as e:
        logger.error(f"vault_read_section error for {path}: {e}")
        return json.dumps({"error": str(e), "path": path})


def vault_read(path: str, force: bool = False, max_chars: int | None = None) -> str:
    """Read a file from the vault, returning content, metadata, and parsed frontmatter."""
    try:
        resolve_vault_path(path)
        content, metadata = read_file(path)

        fm_data = None
        try:
            post = frontmatter.loads(content)
            if post.metadata:
                fm_data = post.metadata
        except Exception:
            pass

        # Check read_policy speed bump
        if fm_data and fm_data.get("read_policy") == "section-only" and not force:
            size_kb = metadata.get("size", 0) // 1024
            return json.dumps(sanitize_for_json({
                "path": path,
                "warning": (
                    f"This file is {size_kb}KB with read_policy: section-only. "
                    f"Use vault_read_section(path, section) for targeted reading, "
                    f"or vault_read(path, force=True) to read the full file."
                ),
                "preview": content[:500],
                "metadata": metadata,
                "frontmatter": fm_data,
            }), cls=SafeJSONEncoder)

        truncated = False
        if max_chars and len(content) > max_chars:
            content = content[:max_chars]
            truncated = True

        result = {
            "path": path,
            "content": content,
            "metadata": metadata,
            "frontmatter": fm_data,
        }
        if truncated:
            result["truncated"] = True
            result["truncated_at"] = max_chars

        return json.dumps(sanitize_for_json(result), cls=SafeJSONEncoder)
    except ValueError as e:
        return json.dumps({"error": str(e), "path": path})
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {path}", "path": path})
    except Exception as e:
        logger.error(f"vault_read error for {path}: {e}")
        return json.dumps({"error": str(e), "path": path})


def vault_batch_read(paths: list[str], include_content: bool = True, force: bool = False) -> str:
    """Read multiple files from the vault in one call."""
    results = []
    found = 0
    missing = 0

    for path in paths:
        try:
            content, metadata = read_file(path)

            fm_data = None
            try:
                post = frontmatter.loads(content)
                if post.metadata:
                    fm_data = post.metadata
            except Exception:
                pass

            # Check read_policy per-file
            if fm_data and fm_data.get("read_policy") == "section-only" and not force:
                size_kb = metadata.get("size", 0) // 1024
                entry = {
                    "path": path,
                    "metadata": metadata,
                    "frontmatter": fm_data,
                    "warning": (
                        f"This file is {size_kb}KB with read_policy: section-only. "
                        f"Use vault_read_section for targeted reading, "
                        f"or pass force=True to read full content."
                    ),
                    "preview": content[:500],
                }
                results.append(entry)
                found += 1
                continue

            entry = {
                "path": path,
                "metadata": metadata,
                "frontmatter": fm_data,
            }
            if include_content:
                entry["content"] = content

            results.append(entry)
            found += 1
        except (ValueError, FileNotFoundError) as e:
            results.append({"path": path, "error": str(e)})
            missing += 1
        except Exception as e:
            results.append({"path": path, "error": str(e)})
            missing += 1

    return json.dumps(sanitize_for_json({"files": results, "found": found, "missing": missing}), cls=SafeJSONEncoder)
