"""Write tools for the Obsidian vault MCP server."""

import json
import logging
import re

import frontmatter

from ..vault import resolve_vault_path, read_file, write_file_atomic, RevisionConflictError, conflict_payload
from ..utils import sanitize_for_json, SafeJSONEncoder

logger = logging.getLogger(__name__)


def vault_write(
    path: str,
    content: str,
    create_dirs: bool = True,
    merge_frontmatter: bool = False,
    expected_revision: str | None = None,
) -> str:
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

        is_new, size = write_file_atomic(
            path, content, create_dirs=create_dirs, tool="vault_write", expected_revision=expected_revision
        )

        return json.dumps({"path": path, "created": is_new, "size": size})
    except RevisionConflictError as e:
        return json.dumps({"error": str(e), "path": path, **conflict_payload(e)})
    except ValueError as e:
        return json.dumps({"error": str(e), "path": path})
    except Exception as e:
        logger.error(f"vault_write error for {path}: {e}")
        return json.dumps({"error": str(e), "path": path})


def vault_batch_frontmatter_update(updates: list[dict]) -> str:
    """Update frontmatter fields on multiple files without changing body content.

    Each update dict may include an optional 'expected_revision' key.
    """
    results = []

    for update in updates:
        file_path = update.get("path", "")
        fields = update.get("fields", {})
        expected_revision = update.get("expected_revision")

        try:
            content, _ = read_file(file_path)
            post = frontmatter.loads(content)

            for key, value in fields.items():
                post.metadata[key] = value

            new_content = frontmatter.dumps(post)
            write_file_atomic(
                file_path,
                new_content,
                create_dirs=False,
                tool="vault_batch_frontmatter_update",
                expected_revision=expected_revision,
            )

            results.append({"path": file_path, "updated": True})
        except FileNotFoundError:
            results.append({"path": file_path, "updated": False, "error": "File not found"})
        except RevisionConflictError as e:
            results.append({"path": file_path, "updated": False, "error": str(e), **conflict_payload(e)})
        except ValueError as e:
            results.append({"path": file_path, "updated": False, "error": str(e)})
        except Exception as e:
            results.append({"path": file_path, "updated": False, "error": str(e)})

    return json.dumps({"results": results})


def _split_leading_heading(text: str) -> tuple[str | None, str]:
    """If `text` begins with a markdown heading line, return (heading, remainder).

    `heading` has surrounding whitespace and any line ending stripped. `remainder`
    is everything after that heading line's line ending (or "" if the heading was
    the only content). Returns (None, text) if `text` does not start with a heading.
    """
    if not text:
        return None, text
    newline_pos = text.find('\n')
    first_line = text if newline_pos == -1 else text[:newline_pos]
    remainder = '' if newline_pos == -1 else text[newline_pos + 1:]
    candidate = first_line.rstrip('\r').strip()
    if not candidate.startswith('#'):
        return None, text
    level = len(candidate) - len(candidate.lstrip('#'))
    if level > 6 or not (len(candidate) > level and candidate[level] == ' '):
        return None, text
    return candidate, remainder


def vault_patch_section(path: str, section: str, content: str, expected_revision: str | None = None) -> str:
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

        # The file's own heading line (lines[target_line]) is always kept as-is.
        # If the caller's content redundantly repeats that heading, drop the
        # caller's copy so the result has exactly one heading. If it leads with
        # a *different* heading, that's ambiguous — refuse rather than guess.
        leading_heading, body_after_heading = _split_leading_heading(content)
        if leading_heading is not None:
            if leading_heading != section_stripped:
                return json.dumps({
                    "error": (
                        f"content starts with heading {leading_heading!r}, which does not match "
                        f"target section {section_stripped!r}. Pass content without a leading "
                        "heading, or with the exact matching heading."
                    ),
                    "path": path,
                })
            replacement = body_after_heading
        else:
            replacement = content

        # Normalise replacement: ensure it ends with a newline if non-empty
        if replacement and not replacement.endswith('\n'):
            replacement += '\n'

        new_content = ''.join(lines[:target_line + 1]) + replacement + ''.join(lines[end_line:])

        _, size = write_file_atomic(
            path, new_content, tool="vault_patch_section", expected_revision=expected_revision
        )
        return json.dumps({"path": path, "section": section, "size": size})
    except FileNotFoundError as e:
        return json.dumps({"error": str(e), "path": path})
    except RevisionConflictError as e:
        return json.dumps({"error": str(e), "path": path, **conflict_payload(e)})
    except ValueError as e:
        return json.dumps({"error": str(e), "path": path})
    except Exception as e:
        logger.error(f"vault_patch_section error for {path}: {e}")
        return json.dumps({"error": str(e), "path": path})


def vault_str_replace(
    path: str, old_str: str, new_str: str, regex: bool = False, expected_revision: str | None = None
) -> str:
    """Replace a unique string in a vault file with another string. Supports regex."""
    try:
        content, _ = read_file(path)
        if regex:
            matches = re.findall(old_str, content)
            if len(matches) == 0:
                return json.dumps({"error": f"Regex pattern not found in {path}", "path": path})
            if len(matches) > 1:
                return json.dumps({
                    "error": f"Regex pattern matches {len(matches)} times in {path} — must be unique. Refine the pattern.",
                    "path": path,
                })
            new_content = re.sub(old_str, new_str, content, count=1)
        else:
            count = content.count(old_str)
            if count == 0:
                return json.dumps({"error": f"String not found in {path}", "path": path})
            if count > 1:
                return json.dumps({
                    "error": f"String appears {count} times in {path} — must be unique. Add surrounding context to disambiguate.",
                    "path": path,
                })
            new_content = content.replace(old_str, new_str, 1)
        _, size = write_file_atomic(
            path, new_content, tool="vault_str_replace", expected_revision=expected_revision
        )
        return json.dumps(sanitize_for_json({
            "path": path,
            "old_length": len(content),
            "new_length": len(new_content),
            "changed": True,
        }), cls=SafeJSONEncoder)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e), "path": path})
    except RevisionConflictError as e:
        return json.dumps({"error": str(e), "path": path, **conflict_payload(e)})
    except ValueError as e:
        return json.dumps({"error": str(e), "path": path})
    except re.error as e:
        return json.dumps({"error": f"Invalid regex: {e}", "path": path})
    except Exception as e:
        logger.error(f"vault_str_replace error for {path}: {e}")
        return json.dumps({"error": str(e), "path": path})


def vault_batch_str_replace(replacements: list[dict]) -> str:
    """Replace unique strings in multiple files in one call.

    Each replacement dict may include an optional 'expected_revision' key.
    """
    results = []
    changed = 0
    failed = 0

    for item in replacements:
        file_path = item.get("path", "")
        old_str = item.get("old_str", "")
        new_str = item.get("new_str", "")
        use_regex = item.get("regex", False)
        expected_revision = item.get("expected_revision")

        try:
            content, _ = read_file(file_path)
            if use_regex:
                matches = re.findall(old_str, content)
                if len(matches) == 0:
                    results.append({"path": file_path, "error": "Regex pattern not found"})
                    failed += 1
                    continue
                if len(matches) > 1:
                    results.append({"path": file_path, "error": f"Regex matches {len(matches)} times — must be unique"})
                    failed += 1
                    continue
                new_content = re.sub(old_str, new_str, content, count=1)
            else:
                count = content.count(old_str)
                if count == 0:
                    results.append({"path": file_path, "error": "String not found"})
                    failed += 1
                    continue
                if count > 1:
                    results.append({"path": file_path, "error": f"String appears {count} times — must be unique"})
                    failed += 1
                    continue
                new_content = content.replace(old_str, new_str, 1)
            write_file_atomic(
                file_path, new_content, tool="vault_batch_str_replace", expected_revision=expected_revision
            )
            results.append({
                "path": file_path,
                "changed": True,
                "old_length": len(content),
                "new_length": len(new_content),
            })
            changed += 1
        except FileNotFoundError:
            results.append({"path": file_path, "error": f"File not found: {file_path}"})
            failed += 1
        except RevisionConflictError as e:
            results.append({"path": file_path, "error": str(e), **conflict_payload(e)})
            failed += 1
        except re.error as e:
            results.append({"path": file_path, "error": f"Invalid regex: {e}"})
            failed += 1
        except Exception as e:
            logger.error(f"vault_batch_str_replace error for {file_path}: {e}")
            results.append({"path": file_path, "error": str(e)})
            failed += 1

    return json.dumps(sanitize_for_json({"results": results, "changed": changed, "failed": failed}), cls=SafeJSONEncoder)


def vault_append(
    path: str, content: str, ensure_newline: bool = True, expected_revision: str | None = None
) -> str:
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

        is_new, size = write_file_atomic(
            path, new_content, create_dirs=True, tool="vault_append", expected_revision=expected_revision
        )
        return json.dumps({
            "path": path,
            "created": is_new,
            "size": size,
            "appended_bytes": len(content.encode("utf-8")),
        })
    except RevisionConflictError as e:
        return json.dumps({"error": str(e), "path": path, **conflict_payload(e)})
    except ValueError as e:
        return json.dumps({"error": str(e), "path": path})
    except Exception as e:
        logger.error(f"vault_append error for {path}: {e}")
        return json.dumps({"error": str(e), "path": path})


def vault_batch_write(files: list[dict]) -> str:
    """Write multiple files in a single call; failures are reported, not raised.

    Each file dict may include an optional 'expected_revision' key.
    """
    results = []
    written = 0
    failed = 0

    for item in files:
        file_path = item.get("path", "")
        file_content = item.get("content", "")
        create_dirs = item.get("create_dirs", True)
        expected_revision = item.get("expected_revision")

        try:
            is_new, size = write_file_atomic(
                file_path,
                file_content,
                create_dirs=create_dirs,
                tool="vault_batch_write",
                expected_revision=expected_revision,
            )
            results.append({"path": file_path, "written": True, "created": is_new, "size": size})
            written += 1
        except RevisionConflictError as e:
            results.append({"path": file_path, "written": False, "error": str(e), **conflict_payload(e)})
            failed += 1
        except (ValueError, OSError) as e:
            results.append({"path": file_path, "written": False, "error": str(e)})
            failed += 1
        except Exception as e:
            logger.error(f"vault_batch_write error for {file_path}: {e}")
            results.append({"path": file_path, "written": False, "error": str(e)})
            failed += 1

    return json.dumps({"results": results, "written": written, "failed": failed})
