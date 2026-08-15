#!/usr/bin/env python3
"""dreaming.py — Nightly report-only maintenance cycle for a vault, plus a
zero-LLM entity index pass.

Cron: 30 15 * * * (01:30 AEST), one invocation per vault via VAULT_PATH env var.
Six report passes: index reconcile, broken wikilink scan, archive candidates,
hot.md budget check, near-duplicate detection, and (Sundays, BS Brain only)
a contradiction-lint pass over infrastructure.md vs. infrastructure-changelog.md.
Plus an entity-index pass that (re)writes `_entities.json` at the vault root.

The report passes remain REPORT-ONLY with respect to vault *content*: existing
.md files are never modified, and the report itself is the only new content
file written per run. `_entities.json` is a second, machine-generated artifact
(not curated vault content) rebuilt from scratch every run — deterministic and
idempotent, so an unchanged vault produces byte-identical output. It, the
semantic index db under .semantic-index/, and the report output are all
excluded from "content" scans.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import frontmatter

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obsidian_vault_mcp import config  # noqa: E402
from obsidian_vault_mcp.frontmatter_safe import (  # noqa: E402
    FrontmatterError,
    update_frontmatter_field,
)
from obsidian_vault_mcp.tools import semantic_search as ss  # noqa: E402

VAULT_PATH = config.VAULT_PATH
VAULT_NAME = VAULT_PATH.name
# generated tool output, not curated vault content — graphify emits synthetic
# `[[...]]`-bracketed community labels that aren't real Obsidian wikilinks
EXCLUDED_DIRS = config.EXCLUDED_DIRS | {"graphify-out"}

HOT_MD_BUDGET_CHARS = 2500
STALE_DAYS = 30
NEAR_DUP_SIMILARITY = 0.93
NEAR_DUP_MAX_WORDS = 300
CONTRADICTION_LOOKBACK_DAYS = 90

ARCHIVE_TYPE_HINTS = {"cc-prompt", "build-log", "cc-summary", "proposed-auto"}
ARCHIVE_STATUS_DONE = {"done", "completed", "complete", "synced", "superseded", "archived"}
ARCHIVE_NAME_HINTS = ("-output.md", "-log.md", "-prompt.md")
ARCHIVE_PATH_HINTS = ("pending-logs", "synced-logs", "build-logs", "build-log")

WIKILINK_RE = re.compile(r"\[\[([^\]\|#]+)")

FENCE_OPEN_RE = re.compile(r"^([ \t]*)(`{3,}|~{3,})")
INDENT_CODE_RE = re.compile(r"^(\t| {4,})\S")
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# Placeholder/identifier link targets that are never real vault paths — template
# examples in _SCHEMA.md/prompt files, or internal snake_case build-log ids.
PLACEHOLDER_TARGETS = {"wikilink", "wikilinks", "target", "entity", "note name", "path/to/file"}
SNAKE_CASE_IDENTIFIER_RE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)+$")


def _strip_non_prose(content: str) -> str:
    """Blank out fenced/indented code blocks and inline code spans so wikilink
    extraction never fires on code syntax (e.g. bash `[[ "$X" == "0" ]]` tests).
    Blanks regions in place (never deletes lines) so line numbers stay accurate."""
    lines = content.split("\n")
    has_fence = any(FENCE_OPEN_RE.match(line) for line in lines)
    out = []
    fence_char = None
    for line in lines:
        if fence_char is not None:
            if re.match(rf"^[ \t]*{re.escape(fence_char)}{{3,}}\s*$", line):
                fence_char = None
            out.append("")
            continue
        m = FENCE_OPEN_RE.match(line)
        if m:
            fence_char = m.group(2)[0]
            out.append("")
            continue
        if not has_fence and INDENT_CODE_RE.match(line):
            out.append("")
            continue
        out.append(INLINE_CODE_RE.sub(lambda mm: " " * len(mm.group(0)), line))
    return "\n".join(out)


def _is_suppressed_link_target(target: str) -> bool:
    """True if `target` is a documentation placeholder or internal identifier,
    never a real vault path, per the dreaming-autofix-and-denoise spec allowlist."""
    t = target.strip()
    tl = t.lower()
    if tl in PLACEHOLDER_TARGETS:
        return True
    if tl.endswith("/path/to/file"):
        return True
    if SNAKE_CASE_IDENTIFIER_RE.match(t):
        return True
    if "..." in t:
        return True
    if t.endswith("/"):
        return True
    return False

# Entity index (zero-LLM): folders whose .md files are always entities, per vault.
# Layout differs per vault (BS: flat Clients/; Alcove: Clients/A-Z/ subfolders;
# CB: no Clients/-style folder — relies entirely on the frontmatter-type fallback
# below). Override per-deployment via ENTITY_FOLDERS_JSON env var
# (e.g. '{"bs-brain": ["BS 2nd Brain/Alcove/Clients"]}').
ENTITY_FOLDERS: dict[str, list[str]] = {
    "bs-brain": [
        "BS 2nd Brain/Alcove/Clients",
        "BS 2nd Brain/Alcove/Team",
        "BS 2nd Brain/Alcove/Referral Partners",
    ],
    "alcove-brain": [
        "Alcove Brain/Clients",
        "Alcove Brain/Team",
        "Alcove Brain/Referral Partners",
    ],
    "cb-brain": [],
}
ENTITY_TYPE_HINTS = {"client", "person", "reference"}
COUPLE_SEP = " & "
MAX_STORED_BACKLINKS = 50


def _is_report_dir(parts: tuple[str, ...]) -> bool:
    """True if `parts` is (or is inside) a dreaming-report output directory.

    Report files echo `[[target]]` syntax when describing broken links, so
    they must be excluded from every pass — otherwise each night's report
    becomes "content" the next night's scan flags against itself.
    """
    if "dreaming-reports" in parts:
        return True
    if "_Reports" in parts:
        idx = parts.index("_Reports")
        if idx + 1 < len(parts) and parts[idx + 1] == "dreaming":
            return True
    return False


def list_md_files(vault_path: Path) -> list[str]:
    """Return sorted vault-relative paths of all .md files, excluding EXCLUDED_DIRS
    and prior dreaming-cycle report output."""
    files = []
    for dirpath, dirnames, filenames in os.walk(vault_path):
        rel_dir = os.path.relpath(dirpath, vault_path)
        parts = Path(rel_dir).parts if rel_dir != "." else ()
        if any(p in EXCLUDED_DIRS for p in parts) or _is_report_dir(parts):
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames if d not in EXCLUDED_DIRS and not _is_report_dir(parts + (d,))
        ]
        for fn in filenames:
            if fn.endswith(".md"):
                rel = fn if rel_dir == "." else os.path.normpath(os.path.join(rel_dir, fn))
                files.append(rel)
    return sorted(files)


def first_h1(content: str) -> str | None:
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _read(vault_path: Path, rel: str) -> str:
    return (vault_path / rel).read_text(encoding="utf-8", errors="replace")


def pass_index_reconcile() -> dict:
    """Reuse the live server's embedding index builder; report what changed."""
    if not ss.SEMANTIC_AVAILABLE:
        return {"status": "skipped", "reason": "fastembed/sqlite-vec not available"}

    db = ss._open_db()
    ss._ensure_schema(db)
    before = {
        row[0]: row[1]
        for row in db.execute("SELECT DISTINCT file_path, mtime FROM chunks").fetchall()
    }
    db.close()

    ss.build_index()

    db = ss._open_db()
    after = {
        row[0]: row[1]
        for row in db.execute("SELECT DISTINCT file_path, mtime FROM chunks").fetchall()
    }
    db.close()

    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(p for p in (set(after) & set(before)) if after[p] != before[p])

    return {
        "status": "ok",
        "indexed_files": len(after),
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def pass_broken_wikilinks(vault_path: Path, md_files: list[str]) -> dict:
    stems = _build_stem_index(md_files)

    broken = []
    suppressed_count = 0
    for rel in md_files:
        content = _strip_non_prose(_read(vault_path, rel))
        for m in WIKILINK_RE.finditer(content):
            target = m.group(1).strip()
            if not target:
                continue
            if _is_suppressed_link_target(target):
                suppressed_count += 1
                continue
            target_stem = Path(target).stem.lower()
            if target_stem not in stems:
                broken.append({"file": rel, "link": target})
    return {"broken": broken, "suppressed_count": suppressed_count}


def _split_person(segment: str) -> tuple[str | None, str]:
    """'Surname, Given' -> (surname, given); 'Given' alone -> (None, given)."""
    if ", " in segment:
        surname, given = segment.split(", ", 1)
        return surname.strip(), given.strip()
    return None, segment.strip()


def generate_aliases(canonical_name: str) -> list[str]:
    """Given-name+surname aliases for a 'Surname, Given[ & Surname2, Given2]' name.

    Couple files where the second person shares the first person's surname omit
    it (e.g. "Duff, Scott & Tracey.md") -- the shared surname is inferred from
    the first segment.
    """
    parts = [p.strip() for p in canonical_name.split(COUPLE_SEP)]
    aliases: list[str] = []

    surname1, given1 = _split_person(parts[0])
    if surname1:
        aliases.append(f"{given1} {surname1}")

    if len(parts) > 1:
        surname2, given2 = _split_person(parts[1])
        surname_for_2 = surname2 or surname1
        if surname_for_2:
            aliases.append(f"{given2} {surname_for_2}")

    return aliases


def _rel_under(rel: str, folder: str) -> bool:
    rel_parts = Path(rel).parts
    folder_parts = Path(folder).parts
    return rel_parts[: len(folder_parts)] == folder_parts


def _entity_folders_for(vault_name: str) -> list[str]:
    override = os.environ.get("ENTITY_FOLDERS_JSON")
    if override:
        try:
            data = json.loads(override)
            folders = data.get(vault_name)
            if folders is not None:
                return list(folders)
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    return ENTITY_FOLDERS.get(vault_name, [])


def _entity_candidates(vault_path: Path, vault_name: str, md_files: list[str]) -> list[str]:
    """Entity folder members, plus any file with frontmatter type in ENTITY_TYPE_HINTS."""
    folders = _entity_folders_for(vault_name)
    candidates: set[str] = set()
    for rel in md_files:
        if any(_rel_under(rel, folder) for folder in folders):
            candidates.add(rel)
            continue
        try:
            post = frontmatter.loads(_read(vault_path, rel))
        except Exception:
            continue
        if str(post.metadata.get("type", "")).lower() in ENTITY_TYPE_HINTS:
            candidates.add(rel)
    return sorted(candidates)


def _whole_word_positions(content: str, needle: str) -> list[int]:
    """Case-insensitive offsets where `needle` occurs with non-alnum (or
    string-edge) boundaries on both sides -- a "whole word" match for
    multi-token names that plain \\b regex boundaries can't handle cleanly
    (names contain commas, apostrophes, etc.)."""
    if not needle:
        return []
    lowered, needle_l = content.lower(), needle.lower()
    hits, start = [], 0
    while True:
        idx = lowered.find(needle_l, start)
        if idx == -1:
            break
        before_ok = idx == 0 or not lowered[idx - 1].isalnum()
        end = idx + len(needle_l)
        after_ok = end >= len(lowered) or not lowered[end].isalnum()
        if before_ok and after_ok:
            hits.append(idx)
        start = idx + 1
    return hits


def _line_for_offset(content: str, offset: int) -> tuple[int, str]:
    line_no = content.count("\n", 0, offset) + 1
    line_start = content.rfind("\n", 0, offset) + 1
    line_end = content.find("\n", offset)
    if line_end == -1:
        line_end = len(content)
    return line_no, content[line_start:line_end].strip()


def _find_backlinks(vault_path: Path, md_files: list[str], entity_rel: str, names: list[str]) -> list[dict]:
    """Every file containing `[[name]]` (any alias) or an exact whole-word
    plain-text match of `name`, with the matching line captured."""
    names_lower = {n.lower() for n in names if n}
    backlinks = []
    for rel in md_files:
        if rel == entity_rel:
            continue
        content = _read(vault_path, rel)
        seen_lines: set[int] = set()

        for m in WIKILINK_RE.finditer(content):
            if m.group(1).strip().lower() in names_lower:
                line_no, line_text = _line_for_offset(content, m.start())
                if line_no not in seen_lines:
                    backlinks.append({"path": rel, "line": line_no, "text": line_text})
                    seen_lines.add(line_no)

        for name in names:
            for offset in _whole_word_positions(content, name):
                line_no, line_text = _line_for_offset(content, offset)
                if line_no not in seen_lines:
                    backlinks.append({"path": rel, "line": line_no, "text": line_text})
                    seen_lines.add(line_no)

    return backlinks


def pass_entity_index(vault_path: Path, vault_name: str, md_files: list[str]) -> list[dict]:
    """Zero-LLM entity index: canonical name (filename sans .md), aliases,
    path, type, and backlinks for every entity-folder member or frontmatter
    type={client,person,reference} file."""
    entities = []
    for rel in _entity_candidates(vault_path, vault_name, md_files):
        canonical = Path(rel).stem

        try:
            post = frontmatter.loads(_read(vault_path, rel))
            fm_aliases = post.metadata.get("aliases") or []
            if isinstance(fm_aliases, str):
                fm_aliases = [fm_aliases]
            ftype = str(post.metadata.get("type", "")).lower() or None
        except Exception:
            fm_aliases, ftype = [], None

        aliases = list(dict.fromkeys([*fm_aliases, *generate_aliases(canonical)]))
        backlinks = _find_backlinks(vault_path, md_files, rel, [canonical, *aliases])

        entities.append({
            "name": canonical,
            "path": rel,
            "type": ftype,
            "aliases": aliases,
            "backlinks": backlinks[:MAX_STORED_BACKLINKS],
            "backlinks_truncated": len(backlinks) > MAX_STORED_BACKLINKS,
        })
    return entities


def write_entities_json(vault_path: Path, vault_name: str, now: datetime, entities: list[dict]) -> Path:
    out_path = vault_path / "_entities.json"
    payload = {
        "vault": vault_name,
        "generated": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entity_count": len(entities),
        "entities": entities,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def pass_archive_candidates(vault_path: Path, md_files: list[str], now: datetime) -> list[dict]:
    candidates = []
    for rel in md_files:
        path = vault_path / rel
        try:
            stat = path.stat()
        except OSError:
            continue
        age_days = (now - datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)).days
        if age_days < STALE_DAYS:
            continue

        try:
            post = frontmatter.loads(_read(vault_path, rel))
        except Exception:
            continue
        ftype = str(post.metadata.get("type", "")).lower()
        status = str(post.metadata.get("status", "")).lower()
        name = Path(rel).name.lower()

        looks_one_shot = (
            ftype in ARCHIVE_TYPE_HINTS
            or name.endswith(ARCHIVE_NAME_HINTS)
            or any(hint in rel.lower() for hint in ARCHIVE_PATH_HINTS)
        )
        completed = status in ARCHIVE_STATUS_DONE

        if looks_one_shot and completed:
            candidates.append({
                "path": rel,
                "age_days": age_days,
                "type": ftype,
                "status": status,
            })
    return candidates


def pass_hot_md_budget(vault_path: Path, md_files: list[str]) -> list[dict]:
    flagged = []
    for rel in md_files:
        if Path(rel).name.lower() != "hot.md":
            continue
        size = len(_read(vault_path, rel))
        if size > HOT_MD_BUDGET_CHARS:
            flagged.append({"path": rel, "chars": size, "budget": HOT_MD_BUDGET_CHARS})
    return flagged


BO_SPECS_DIR = "Personal/Build Orchestrator/specs"
BO_BUILD_LOGS_DIR = "Personal/Build Orchestrator/build-logs"
BO_AUTO_DATE_PREFIX_RE = re.compile(r"^auto-\d{8}-")
BO_AUTO_DATE_INSTRUMENT_PREFIX_RE = re.compile(r"^auto-\d{4}-\d{2}-\d{2}-[A-Za-z]+-")
RETRIEVAL_EVAL_REPORT_RE = re.compile(r"retrieval-eval[/\\][^/\\]*-report\.md$", re.IGNORECASE)
SUSPECT_NEAR_DUP_SIMILARITY = 0.999


def _normalize_bo_stem(stem: str) -> str:
    """Strip the BO trailing '-output' and either date-prefix convention
    ('auto-YYYYMMDD-' or 'auto-YYYY-MM-DD-{INSTRUMENT}-') so a spec's stem and
    its build-log's stem compare equal for the same build id."""
    s = stem
    if s.endswith("-output"):
        s = s[: -len("-output")]
    s = BO_AUTO_DATE_INSTRUMENT_PREFIX_RE.sub("", s)
    s = BO_AUTO_DATE_PREFIX_RE.sub("", s)
    return s


def _is_structural_bo_pair(paths: list[str]) -> bool:
    """True if `paths` is exactly one Build Orchestrator spec + its build-log,
    for the same build id — the designed BO spec/build-log convention, not a
    real duplicate."""
    if len(paths) != 2:
        return False
    specs = [p for p in paths if _rel_under(p, BO_SPECS_DIR)]
    logs = [p for p in paths if _rel_under(p, BO_BUILD_LOGS_DIR)]
    if len(specs) != 1 or len(logs) != 1:
        return False
    return _normalize_bo_stem(Path(specs[0]).stem) == _normalize_bo_stem(Path(logs[0]).stem)


def _is_retrieval_eval_report(path: str) -> bool:
    return bool(RETRIEVAL_EVAL_REPORT_RE.search(path))


def pass_near_duplicates(vault_path: Path, md_files: list[str]) -> dict:
    titles: dict[str, list[str]] = {}
    for rel in md_files:
        content = _read(vault_path, rel)
        title = (first_h1(content) or Path(rel).stem).strip().lower()
        titles.setdefault(title, []).append(rel)
    title_matches = [
        {"title": t, "files": paths}
        for t, paths in sorted(titles.items())
        if len(paths) > 1 and not _is_structural_bo_pair(paths)
    ]

    embedding_matches: list[dict] = []
    suspect_lines: list[str] = []
    if ss.SEMANTIC_AVAILABLE and len(md_files) > 1:
        import numpy as np

        model = ss._get_model()
        texts = []
        for rel in md_files:
            words = _read(vault_path, rel).split()
            texts.append(" ".join(words[:NEAR_DUP_MAX_WORDS]))

        embeddings = list(model.embed(texts))
        arr = np.array(embeddings)
        norms = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        sim = norms @ norms.T

        n = len(md_files)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim[i, j])
                if score <= NEAR_DUP_SIMILARITY:
                    continue
                a, b = md_files[i], md_files[j]
                if _is_retrieval_eval_report(a) and _is_retrieval_eval_report(b):
                    # successive eval reports are expected to be near-identical
                    # by construction; suppress, but flag suspiciously exact ones
                    if score >= SUSPECT_NEAR_DUP_SIMILARITY:
                        suspect_lines.append(f"`{a}` ↔ `{b}` (similarity {round(score, 4)})")
                    continue
                embedding_matches.append({
                    "a": a,
                    "b": b,
                    "similarity": round(score, 4),
                })

    return {
        "title_matches": title_matches,
        "embedding_matches": embedding_matches,
        "suspect_lines": suspect_lines,
    }


def _split_changelog_entries(text: str) -> list[tuple[str | None, str]]:
    """Split a changelog on '## ' headers; return (date-or-None, entry-text) pairs."""
    entries = []
    date_re = re.compile(r"^(\d{4}-\d{2}-\d{2})")
    for block in re.split(r"\n(?=## )", text):
        block = block.strip()
        if not block:
            continue
        header = block[3:].splitlines()[0] if block.startswith("## ") else ""
        m = date_re.match(header.strip())
        entries.append((m.group(1) if m else None, block))
    return entries


def pass_contradiction_lint_sunday(vault_path: Path, vault_name: str, now: datetime) -> dict | None:
    if vault_name != "bs-brain" or now.weekday() != 6:
        return None

    infra_path = vault_path / "BS 2nd Brain" / "Alcove" / "Infrastructure" / "infrastructure.md"
    changelog_path = vault_path / "BS 2nd Brain" / "Alcove" / "Infrastructure" / "infrastructure-changelog.md"
    if not infra_path.exists() or not changelog_path.exists():
        return {"status": "skipped", "reason": "infrastructure.md or infrastructure-changelog.md not found"}

    cutoff = now - timedelta(days=CONTRADICTION_LOOKBACK_DAYS)
    entries = _split_changelog_entries(changelog_path.read_text(encoding="utf-8", errors="replace"))
    recent_entries = [
        text for date_str, text in entries
        if date_str and datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) >= cutoff
    ]

    claim_re = re.compile(r"(port\s*\d{3,5}|:\d{3,5}\b|/home/\S+|https?://\S+)", re.IGNORECASE)
    infra_lines = infra_path.read_text(encoding="utf-8", errors="replace").splitlines()

    candidates = []
    for i, line in enumerate(infra_lines):
        for m in claim_re.finditer(line):
            claim = m.group(0).strip()
            for entry in recent_entries:
                if claim.lower() in entry.lower():
                    candidates.append({
                        "infra_line": i + 1,
                        "claim": line.strip()[:200],
                        "changelog_excerpt": entry.strip().splitlines()[0][:200],
                    })
                    break

    return {
        "status": "ok",
        "recent_changelog_entries_scanned": len(recent_entries),
        "candidate_contradictions": candidates[:50],
    }


def build_report(
    vault_name: str,
    now: datetime,
    reconcile: dict,
    broken_links_result: dict,
    archive_candidates: list[dict],
    hot_md_flags: list[dict],
    near_dups: dict,
    contradiction: dict | None,
) -> str:
    broken_links = broken_links_result["broken"]
    suppressed_count = broken_links_result["suppressed_count"]
    date_str = now.strftime("%Y-%m-%d")
    lines = [
        "---",
        "build_id: vault-dreaming-cycle",
        f"vault: {vault_name}",
        f"generated: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "---",
        "",
        f"# Dreaming Cycle Report — {vault_name} — {date_str}",
        "",
        "## What this means",
        "Nightly report-only maintenance scan of this vault. Nothing was edited "
        "— every finding below is a proposal for manual review and action.",
        "",
        "## 1. Index reconcile",
    ]
    if reconcile.get("status") == "skipped":
        lines.append(f"Skipped: {reconcile.get('reason')}")
    else:
        lines.append(f"Indexed files: {reconcile['indexed_files']}")
        lines.append(f"Newly indexed: {len(reconcile['added'])}")
        lines.append(f"Re-embedded (changed): {len(reconcile['changed'])}")
        lines.append(f"Purged (deleted from vault): {len(reconcile['removed'])}")
        if reconcile["removed"]:
            lines.append("")
            lines.append("Purged paths:")
            for p in reconcile["removed"][:20]:
                lines.append(f"- `{p}`")

    lines += ["", "## 2. Broken wikilinks"]
    if broken_links:
        lines.append(f"{len(broken_links)} broken link(s) found:")
        for b in broken_links[:50]:
            lines.append(f"- `{b['file']}` → `[[{b['link']}]]`")
    else:
        lines.append("None found.")
    lines.append(
        f"Suppressed {suppressed_count} link(s) as code-fence/inline-code content, "
        "documentation placeholders, or snake_case internal identifiers."
    )

    lines += ["", "## 3. Archive candidates"]
    if archive_candidates:
        lines.append(f"{len(archive_candidates)} candidate(s) (>{STALE_DAYS}d stale, completion status set):")
        for c in archive_candidates[:50]:
            lines.append(f"- `{c['path']}` — {c['age_days']}d old, type={c['type'] or 'n/a'}, status={c['status']}")
    else:
        lines.append("None found.")

    lines += ["", "## 4. hot.md budget"]
    if hot_md_flags:
        for f in hot_md_flags:
            lines.append(f"- `{f['path']}` — {f['chars']} chars (budget {f['budget']})")
    else:
        lines.append(f"All hot.md files under the {HOT_MD_BUDGET_CHARS}-char budget.")

    lines += ["", "## 5. Near-duplicate detection"]
    title_matches = near_dups.get("title_matches", [])
    embedding_matches = near_dups.get("embedding_matches", [])
    if title_matches:
        lines.append("Same-title matches:")
        for tm in title_matches[:20]:
            lines.append(f"- \"{tm['title']}\": {', '.join(f'`{p}`' for p in tm['files'])}")
    else:
        lines.append("No same-title matches.")
    if embedding_matches:
        lines.append("")
        lines.append(f"Embedding near-duplicates (cosine > {NEAR_DUP_SIMILARITY}):")
        for em in embedding_matches[:20]:
            lines.append(f"- `{em['a']}` ↔ `{em['b']}` (similarity {em['similarity']})")
    elif ss.SEMANTIC_AVAILABLE:
        lines.append(f"No embedding near-duplicates above {NEAR_DUP_SIMILARITY}.")

    suspect_lines = near_dups.get("suspect_lines", [])
    if suspect_lines:
        lines.append("")
        lines.append(f"SUSPECT: eval harness may not be re-running — {'; '.join(suspect_lines)}")

    if contradiction is not None:
        lines += ["", "## 6. Contradiction lint (Sundays only, BS Brain)"]
        if contradiction.get("status") == "skipped":
            lines.append(f"Skipped: {contradiction.get('reason')}")
        else:
            lines.append(
                f"Scanned {contradiction['recent_changelog_entries_scanned']} changelog entries "
                f"from the last {CONTRADICTION_LOOKBACK_DAYS} days."
            )
            candidates = contradiction["candidate_contradictions"]
            if candidates:
                lines.append(f"{len(candidates)} candidate contradiction(s) for manual review:")
                for c in candidates:
                    lines.append(
                        f"- infrastructure.md:{c['infra_line']} `{c['claim']}` "
                        f"vs. changelog: \"{c['changelog_excerpt']}\""
                    )
            else:
                lines.append("No candidate contradictions found.")

    lines += ["", "## Proposed actions"]
    action_count = 0
    for b in broken_links[:20]:
        lines.append(f"- [ ] Fix broken link in `{b['file']}`: `[[{b['link']}]]`")
        action_count += 1
    for c in archive_candidates[:20]:
        lines.append(f"- [ ] Consider archiving `{c['path']}` ({c['age_days']}d old, {c['status']})")
        action_count += 1
    for f in hot_md_flags:
        lines.append(f"- [ ] Trim `{f['path']}` ({f['chars']} chars, budget {HOT_MD_BUDGET_CHARS})")
        action_count += 1
    for tm in title_matches[:10]:
        lines.append(f"- [ ] Review same-title notes: {', '.join(f'`{p}`' for p in tm['files'])}")
        action_count += 1
    for em in embedding_matches[:10]:
        lines.append(f"- [ ] Review near-duplicate: `{em['a']}` vs `{em['b']}`")
        action_count += 1
    if action_count == 0:
        lines.append("- [ ] Nothing to action tonight.")

    return "\n".join(lines) + "\n"


ARCHIVE_EXCLUDED_DIRS = {"_Archive", ".trash"}


def _build_stem_index(md_files: list[str]) -> dict[str, list[str]]:
    stems: dict[str, list[str]] = {}
    for rel in md_files:
        stems.setdefault(Path(rel).stem.lower(), []).append(rel)
    return stems


def _normalize_basename(s: str) -> str:
    """Fold a filename stem down to bare alphanumerics so 'Old Page', 'old-page'
    and 'Old_Page' all compare equal -- the 'mechanical' repair class this
    build's autofix targets (punctuation/case drift), not semantic renames."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_autofix_candidates(vault_path: Path, md_files: list[str]) -> list[dict]:
    """Broken wikilinks (per pass_broken_wikilinks' resolution rule) whose target
    basename, once punctuation/case-folded, matches exactly one vault file --
    an unambiguous mechanical fix. Placeholder/identifier targets (Step 3) are
    never candidates. Links inside _Archive/ or .trash/ files are never touched."""
    stems = _build_stem_index(md_files)
    norm_index: dict[str, list[str]] = {}
    for stem_key, rels in stems.items():
        norm_index.setdefault(_normalize_basename(stem_key), []).extend(rels)

    candidates = []
    for rel in md_files:
        if any(part in ARCHIVE_EXCLUDED_DIRS for part in Path(rel).parts):
            continue
        stripped = _strip_non_prose(_read(vault_path, rel))
        for m in WIKILINK_RE.finditer(stripped):
            target = m.group(1).strip()
            if not target or _is_suppressed_link_target(target):
                continue
            target_stem = Path(target).stem.lower()
            if target_stem in stems:
                continue  # already resolves -- not broken

            matches = sorted(set(norm_index.get(_normalize_basename(Path(target).stem), [])))
            if len(matches) != 1:
                continue
            match_rel = matches[0]
            new_target = match_rel[:-3] if match_rel.lower().endswith(".md") else match_rel

            line_start = stripped.rfind("\n", 0, m.start(1)) + 1
            candidates.append({
                "file": rel,
                "line": stripped.count("\n", 0, m.start(1)) + 1,
                "col": m.start(1) - line_start,
                "old_target": m.group(1),
                "new_target": new_target,
            })
    return candidates


def _bump_frontmatter_updated(content: str, today: str) -> str:
    """Parse and update an existing ``updated`` field."""
    return update_frontmatter_field(content, "updated", today, require_existing=True)


def apply_autofix(vault_path: Path, candidates: list[dict], now: datetime) -> tuple[list[str], Path | None]:
    """Backs up and repairs every candidate. Idempotent: once fixed, a target
    resolves via _build_stem_index and will never be a candidate again."""
    if not candidates:
        return [], None

    backup_dir = Path.home() / "backups" / "dreaming-autofix" / now.strftime("%Y%m%dT%H%M%SZ")
    today = now.strftime("%Y-%m-%d")

    by_file: dict[str, list[dict]] = {}
    for c in candidates:
        by_file.setdefault(c["file"], []).append(c)

    fixed_lines = []
    for rel, cands in by_file.items():
        path = vault_path / rel
        original_content = path.read_text(encoding="utf-8", errors="replace")
        lines = original_content.split("\n")
        file_fixed = []
        for c in sorted(cands, key=lambda c: (c["line"], -c["col"])):
            ln, col, old = c["line"] - 1, c["col"], c["old_target"]
            line_text = lines[ln]
            if line_text[col : col + len(old)] != old:
                continue  # content shifted since scan -- skip rather than corrupt
            lines[ln] = line_text[:col] + c["new_target"] + line_text[col + len(old) :]
            file_fixed.append(f"FIXED: {rel}:{c['line']} [[{old}]] → [[{c['new_target']}]]")

        if not file_fixed:
            continue
        try:
            new_content = _bump_frontmatter_updated("\n".join(lines), today)
        except FrontmatterError as exc:
            print(
                f"[dreaming] SKIPPED {rel}: invalid frontmatter: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue

        backup_path = backup_dir / rel
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(original_content, encoding="utf-8")
        path.write_text(new_content, encoding="utf-8")
        fixed_lines.extend(file_fixed)

    return fixed_lines, (backup_dir if fixed_lines else None)


def report_path_for(vault_path: Path, vault_name: str, now: datetime) -> Path:
    date_str = now.strftime("%Y-%m-%d")
    if vault_name == "bs-brain":
        report_dir = vault_path / "BS 2nd Brain" / "Alcove" / "Infrastructure" / "dreaming-reports"
    else:
        report_dir = vault_path / "_Reports" / "dreaming"
    return report_dir / f"{date_str}.md"


def run(autofix: bool = False) -> Path:
    now = datetime.now(timezone.utc)
    md_files = list_md_files(VAULT_PATH)

    reconcile = pass_index_reconcile()
    broken_links = pass_broken_wikilinks(VAULT_PATH, md_files)
    archive_candidates = pass_archive_candidates(VAULT_PATH, md_files, now)
    hot_md_flags = pass_hot_md_budget(VAULT_PATH, md_files)
    near_dups = pass_near_duplicates(VAULT_PATH, md_files)
    contradiction = pass_contradiction_lint_sunday(VAULT_PATH, VAULT_NAME, now)
    entities = pass_entity_index(VAULT_PATH, VAULT_NAME, md_files)

    report = build_report(
        VAULT_NAME, now, reconcile, broken_links, archive_candidates, hot_md_flags, near_dups, contradiction
    )

    out_path = report_path_for(VAULT_PATH, VAULT_NAME, now)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    entities_path = write_entities_json(VAULT_PATH, VAULT_NAME, now, entities)

    print(
        f"[dreaming] {VAULT_NAME} {now.strftime('%Y-%m-%d %H:%M')} UTC — "
        f"{len(md_files)} files scanned, {len(broken_links['broken'])} broken links, "
        f"{len(archive_candidates)} archive candidates, "
        f"{len(entities)} entities → {out_path}, {entities_path}",
        flush=True,
    )

    if autofix:
        candidates = find_autofix_candidates(VAULT_PATH, md_files)
        fixed_lines, backup_dir = apply_autofix(VAULT_PATH, candidates, now)
        if fixed_lines:
            print(f"[dreaming] --autofix: backed up originals to {backup_dir}", flush=True)
            for line in fixed_lines:
                print(line, flush=True)
        else:
            print("[dreaming] --autofix: no unambiguous mechanical repairs found", flush=True)

    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--autofix",
        action="store_true",
        help="Repair broken wikilinks with exactly one unambiguous basename match",
    )
    cli_args = parser.parse_args()
    run(autofix=cli_args.autofix)
