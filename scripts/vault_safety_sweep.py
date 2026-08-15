#!/usr/bin/env python3
"""Detect and narrowly repair vault frontmatter corruption and unreadable modes.

Dry-run is the default. Content and permission repair are independently gated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obsidian_vault_mcp.frontmatter_safe import (  # noqa: E402
    FrontmatterError,
    parse_frontmatter,
)


DEFAULT_VAULTS = {
    "bs-brain": Path("/home/ben_sum/vaults/bs-brain"),
    "alcove-brain": Path("/home/ben_sum/vaults/alcove-brain"),
    "cb-brain": Path("/home/ben_sum/vaults/cb-brain"),
}
EXCLUDED_DIRS = {".git", ".trash", ".obsidian", ".semantic-index"}
SCALAR_LINE_RE = re.compile(rb"^[A-Za-z_][A-Za-z0-9_-]*:[ \t]+(.+)$")


def _read_fd(fd: int) -> bytes:
    chunks = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _strict_parse_bytes(data: bytes):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FrontmatterError(f"invalid UTF-8: {exc}") from exc
    return parse_frontmatter(text)


def known_signature_repair(data: bytes) -> tuple[bytes, dict] | None:
    """Return the sole safe one-newline repair for the known glued delimiter."""
    if not (data.startswith(b"---\n") or data.startswith(b"---\r\n")):
        return None
    opening_newline = b"\r\n" if data.startswith(b"---\r\n") else b"\n"
    candidates = []
    offset = 0
    for index, line in enumerate(data.splitlines(keepends=True)):
        line_body = line.rstrip(b"\r\n")
        if index == 0:
            offset += len(line)
            continue
        if line_body == b"---":
            if not candidates:
                return None
            break
        if line_body.endswith(b"---"):
            prefix = line_body[:-3]
            match = SCALAR_LINE_RE.fullmatch(prefix)
            value = match.group(1).lstrip() if match else b""
            if match and value and not value.startswith((b"|", b">", b"#")):
                insert_at = offset + len(prefix)
                repaired = data[:insert_at] + opening_newline + data[insert_at:]
                try:
                    document = _strict_parse_bytes(repaired)
                except FrontmatterError:
                    document = None
                if document is not None:
                    candidates.append((repaired, insert_at))
        offset += len(line)

    if len(candidates) != 1:
        return None
    repaired, insert_at = candidates[0]
    before_start = max(0, insert_at - 80)
    after_end = min(len(repaired), insert_at + len(opening_newline) + 80)
    return repaired, {
        "byte_offset": insert_at,
        "before": repr(data[before_start:min(len(data), insert_at + 80)]),
        "after": repr(repaired[before_start:after_end]),
    }


def _atomic_replace_if_unchanged(
    path: Path,
    original_stat,
    original_data: bytes,
    data: bytes,
    mode: int,
    xattrs: dict[str, bytes],
) -> str:
    """Replace only if the scanned regular file still has the same identity."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open(path, os.O_RDONLY | nofollow)
    try:
        current = os.fstat(current_fd)
        current_data = _read_fd(current_fd)
        if not stat.S_ISREG(current.st_mode) or (
            current.st_dev,
            current.st_ino,
        ) != (original_stat.st_dev, original_stat.st_ino):
            raise RuntimeError("identity changed before content repair; refused")
        if current_data != original_data:
            raise RuntimeError("content changed before content repair; refused")
        if stat.S_IMODE(current.st_mode) != mode:
            raise RuntimeError("mode changed before content repair; refused")
    finally:
        os.close(current_fd)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".vault-safety.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            for name, value in xattrs.items():
                os.setxattr(handle.fileno(), name, value)
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return "committed"


def scan_vault(
    name: str,
    root: Path,
    *,
    repair_content: bool = False,
    repair_permissions: bool = False,
) -> dict:
    report = {
        "vault": name,
        "root": str(root),
        "files_scanned": 0,
        "frontmatter_errors": [],
        "permission_errors": [],
        "repairs": [],
        "symlinks": [],
        "operational_errors": [],
    }
    nofollow = getattr(os, "O_NOFOLLOW", 0)

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        kept_dirs = []
        for dirname in dirnames:
            candidate = Path(dirpath) / dirname
            if dirname in EXCLUDED_DIRS:
                continue
            if candidate.is_symlink():
                report["symlinks"].append(str(candidate.relative_to(root)))
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            path = Path(dirpath) / filename
            relative = str(path.relative_to(root))
            try:
                lst = path.lstat()
                if stat.S_ISLNK(lst.st_mode):
                    report["symlinks"].append(relative)
                    continue
                if not stat.S_ISREG(lst.st_mode):
                    continue
                fd = os.open(path, os.O_RDONLY | nofollow)
                try:
                    opened = os.fstat(fd)
                    if not stat.S_ISREG(opened.st_mode):
                        raise RuntimeError("opened candidate is not a regular file")
                    data = _read_fd(fd)
                    xattrs = {
                        name: os.getxattr(fd, name)
                        for name in os.listxattr(fd)
                    }
                    report["files_scanned"] += 1
                    mode = stat.S_IMODE(opened.st_mode)
                    if not mode & stat.S_IROTH:
                        permission_item = {"path": relative, "mode": oct(mode)}
                        report["permission_errors"].append(permission_item)
                        if repair_permissions:
                            new_mode = mode | stat.S_IROTH
                            os.fchmod(fd, new_mode)
                            mode = new_mode
                            report["repairs"].append(
                                {
                                    "path": relative,
                                    "type": "permission",
                                    "before": permission_item["mode"],
                                    "after": oct(new_mode),
                                    "verified": stat.S_IMODE(os.fstat(fd).st_mode) == new_mode,
                                }
                            )
                finally:
                    os.close(fd)

                if path.suffix.lower() != ".md" or not (
                    data.startswith(b"---\n")
                    or data.startswith(b"---\r\n")
                    or data.startswith(b"\xef\xbb\xbf---")
                ):
                    continue
                try:
                    _strict_parse_bytes(data)
                except FrontmatterError as exc:
                    repair = known_signature_repair(data)
                    item = {
                        "path": relative,
                        "error": str(exc),
                        "known_signature": repair is not None,
                    }
                    report["frontmatter_errors"].append(item)
                    if repair_content and repair is not None:
                        repaired, evidence = repair
                        _strict_parse_bytes(repaired)
                        outcome = _atomic_replace_if_unchanged(
                            path, opened, data, repaired, mode, xattrs
                        )
                        verified_data = path.read_bytes()
                        _strict_parse_bytes(verified_data)
                        report["repairs"].append(
                            {
                                "path": relative,
                                "type": "frontmatter",
                                **evidence,
                                "outcome": outcome,
                                "verified": verified_data == repaired,
                            }
                        )
            except Exception as exc:
                report["operational_errors"].append({"path": relative, "error": str(exc)})
    return report


def run(
    vaults: dict[str, Path] | None = None,
    *,
    repair_content: bool = False,
    repair_permissions: bool = False,
) -> dict:
    vaults = vaults or DEFAULT_VAULTS
    reports = [
        scan_vault(
            name,
            root,
            repair_content=repair_content,
            repair_permissions=repair_permissions,
        )
        for name, root in vaults.items()
    ]
    return {
        "mode": "repair" if repair_content or repair_permissions else "dry-run",
        "repair_content": repair_content,
        "repair_permissions": repair_permissions,
        "vaults": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-content", action="store_true")
    parser.add_argument("--repair-permissions", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        repair_content=args.repair_content,
        repair_permissions=args.repair_permissions,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not any(vault["operational_errors"] for vault in report["vaults"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
