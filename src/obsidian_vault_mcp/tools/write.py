"""Write tools for the Obsidian vault MCP server."""

import json
import logging

import frontmatter

from ..vault import resolve_vault_path, read_file, write_file_atomic
from ..utils import sanitize_for_json, SafeJSONEncoder

logger = logging.getLogger(__name__)


def vault_write(path: str, content: str, create_dirs: bool = True, merge_frontmatter: bool = False) -> str:
    """Write a file to the vault, optionally merging frontmatter with existing content."""
    try:
        resolve_vault_path(path)

        if merge_frontmatter:
            try:
                existing_content, _ = read_file(path)
                existing_post = frontmatter.loads(existing_content)
                new_post = frontmatter.loads(content)

                merged_meta = dict(existing_post.metadata)
                merged_meta.update(new_post.metadata)

                new_post.metadata = merged_meta
                content = frontmatter.dumps(new_post)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Frontmatter merge failed for {path}, writing as-is: {e}")

        is_new, size = write_file_atomic(path, content, create_dirs=create_dirs)

        return json.dumps({"path": path, "created": is_new, "size": size})
    except ValueError as e:
        return json.dumps({"error": str(e), "path": path})
    except Exception as e:
        logger.error(f"vault_write error for {path}: {e}")
        return json.dumps({"error": str(e), "path": path})


def vault_batch_frontmatter_update(updates: list[dict]) -> str:
    """Update frontmatter fields on multiple files without changing body content."""
    results = []

    for update in updates:
        file_path = update.get("path", "")
        fields = update.get("fields", {})

        try:
            content, _ = read_file(file_path)
            post = frontmatter.loads(content)

            for key, value in fields.items():
                post.metadata[key] = value

            new_content = frontmatter.dumps(post)
            write_file_atomic(file_path, new_content, create_dirs=False)

            results.append({"path": file_path, "updated": True})
        except FileNotFoundError:
            results.append({"path": file_path, "updated": False, "error": "File not found"})
        except ValueError as e:
            results.append({"path": file_path, "updated": False, "error": str(e)})
        except Exception as e:
            results.append({"path": file_path, "updated": False, "error": str(e)})

    return json.dumps({"results": results})


def vault_patch_section(path: str, section: str, content: str) -> str:
    """Replace the content of a single markdown section without rewriting the entire file."""
    try:
        file_content, _ = read_file(path)
        lines = file_content.splitlines(keepends=True)

        section_stripped = section.strip()
        if not section_stripped.startswith('#'):
            return json.dumps({"error": "section must start with # characters", "path": path})
        heading_level = len(section_stripped) - len(section_stripped.lstrip('#'))
        if heading_level > 6 or not (len(section_stripped) > heading_level and section_stripped[heading_level] == ' '):
            return json.dumps({"error": f"Invalid heading format: {section!r}", "path": path})

        # Find the target heading line (exact match, ignoring line endings)
        target_line = -1
        for i, line in enumerate(lines):
            if line.rstrip('\r\n') == section_stripped:
                target_line = i
                break

        if target_line == -1:
            return json.dumps({"error": f"Heading not found: {section!r}", "path": path})

        # Find end of section: next heading of same or higher level (lower or equal # count)
        end_line = len(lines)
        for i in range(target_line + 1, len(lines)):
            candidate = lines[i].rstrip('\r\n')
            if candidate.startswith('#'):
                level = len(candidate) - len(candidate.lstrip('#'))
                if 1 <= level <= heading_level and len(candidate) > level and candidate[level] == ' ':
                    end_line = i
                    break

        # Normalise replacement: ensure it ends with a newline if non-empty
        replacement = content
        if replacement and not replacement.endswith('\n'):
            replacement += '\n'

        new_content = ''.join(lines[:target_line + 1]) + replacement + ''.join(lines[end_line:])

        _, size = write_file_atomic(path, new_content)
        return json.dumps({"path": path, "section": section, "size": size})
    except FileNotFoundError as e:
        return json.dumps({"error": str(e), "path": path})
    except ValueError as e:
        return json.dumps({"error": str(e), "path": path})
    except Exception as e:
        logger.error(f"vault_patch_section error for {path}: {e}")
        return json.dumps({"error": str(e), "path": path})


def vault_str_replace(path: str, old_str: str, new_str: str) -> str:
    """Replace a unique string in a vault file with another string."""
    try:
        content, _ = read_file(path)
        count = content.count(old_str)
        if count == 0:
            return json.dumps({"error": f"String not found in {path}", "path": path})
        if count > 1:
            return json.dumps({
                "error": f"String appears {count} times in {path} — must be unique. Add surrounding context to disambiguate.",
                "path": path,
            })
        new_content = content.replace(old_str, new_str, 1)
        _, size = write_file_atomic(path, new_content)
        return json.dumps(sanitize_for_json({
            "path": path,
            "old_length": len(content),
            "new_length": len(new_content),
            "changed": True,
        }), cls=SafeJSONEncoder)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e), "path": path})
    except ValueError as e:
        return json.dumps({"error": str(e), "path": path})
    except Exception as e:
        logger.error(f"vault_str_replace error for {path}: {e}")
        return json.dumps({"error": str(e), "path": path})


def vault_append(path: str, content: str, ensure_newline: bool = True) -> str:
    """Append content to an existing vault file, creating it if absent."""
    try:
        try:
            existing_content, _ = read_file(path)
        except FileNotFoundError:
            existing_content = ""

        if ensure_newline and existing_content:
            # Guarantee a blank-line separator before appended content
            NL = chr(10)
            trailing_nls = len(existing_content) - len(existing_content.rstrip(NL))
            if trailing_nls == 0:
                existing_content += NL + NL
            elif trailing_nls == 1:
                existing_content += NL
            # else: already has blank line separator

        new_content = existing_content + content

        is_new, size = write_file_atomic(path, new_content, create_dirs=True)
        return json.dumps({
            "path": path,
            "created": is_new,
            "size": size,
            "appended_bytes": len(content.encode("utf-8")),
        })
    except ValueError as e:
        return json.dumps({"error": str(e), "path": path})
    except Exception as e:
        logger.error(f"vault_append error for {path}: {e}")
        return json.dumps({"error": str(e), "path": path})


def vault_batch_write(files: list[dict]) -> str:
    """Write multiple files in a single call; failures are reported, not raised."""
    results = []
    written = 0
    failed = 0

    for item in files:
        file_path = item.get("path", "")
        file_content = item.get("content", "")
        create_dirs = item.get("create_dirs", True)

        try:
            is_new, size = write_file_atomic(file_path, file_content, create_dirs=create_dirs)
            results.append({"path": file_path, "written": True, "created": is_new, "size": size})
            written += 1
        except (ValueError, OSError) as e:
            results.append({"path": file_path, "written": False, "error": str(e)})
            failed += 1
        except Exception as e:
            logger.error(f"vault_batch_write error for {file_path}: {e}")
            results.append({"path": file_path, "written": False, "error": str(e)})
            failed += 1

    return json.dumps({"results": results, "written": written, "failed": failed})
