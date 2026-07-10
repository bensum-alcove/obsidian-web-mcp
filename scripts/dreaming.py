#!/usr/bin/env python3
"""dreaming.py — Nightly report-only maintenance cycle for a vault.

Cron: 30 15 * * * (01:30 AEST), one invocation per vault via VAULT_PATH env var.
Six passes: index reconcile, broken wikilink scan, archive candidates,
hot.md budget check, near-duplicate detection, and (Sundays, BS Brain only)
a contradiction-lint pass over infrastructure.md vs. infrastructure-changelog.md.

v1 is strictly REPORT-ONLY: it writes exactly one new file (the report) and
touches nothing else in the vault content tree. The semantic index db under
.semantic-index/ is refreshed as part of the index-reconcile pass, but that
directory is excluded from vault "content" and from the report-only guarantee.
"""
from __future__ import annotations

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


def pass_broken_wikilinks(vault_path: Path, md_files: list[str]) -> list[dict]:
    stems: dict[str, list[str]] = {}
    for rel in md_files:
        stems.setdefault(Path(rel).stem.lower(), []).append(rel)

    broken = []
    for rel in md_files:
        content = _read(vault_path, rel)
        for m in WIKILINK_RE.finditer(content):
            target = m.group(1).strip()
            if not target or target.endswith("..."):
                continue
            target_stem = Path(target).stem.lower()
            if target_stem not in stems:
                broken.append({"file": rel, "link": target})
    return broken


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


def pass_near_duplicates(vault_path: Path, md_files: list[str]) -> dict:
    titles: dict[str, list[str]] = {}
    for rel in md_files:
        content = _read(vault_path, rel)
        title = (first_h1(content) or Path(rel).stem).strip().lower()
        titles.setdefault(title, []).append(rel)
    title_matches = [
        {"title": t, "files": paths} for t, paths in sorted(titles.items()) if len(paths) > 1
    ]

    embedding_matches: list[dict] = []
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
                if score > NEAR_DUP_SIMILARITY:
                    embedding_matches.append({
                        "a": md_files[i],
                        "b": md_files[j],
                        "similarity": round(score, 4),
                    })

    return {"title_matches": title_matches, "embedding_matches": embedding_matches}


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
    broken_links: list[dict],
    archive_candidates: list[dict],
    hot_md_flags: list[dict],
    near_dups: dict,
    contradiction: dict | None,
) -> str:
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


def report_path_for(vault_path: Path, vault_name: str, now: datetime) -> Path:
    date_str = now.strftime("%Y-%m-%d")
    if vault_name == "bs-brain":
        report_dir = vault_path / "BS 2nd Brain" / "Alcove" / "Infrastructure" / "dreaming-reports"
    else:
        report_dir = vault_path / "_Reports" / "dreaming"
    return report_dir / f"{date_str}.md"


def run() -> Path:
    now = datetime.now(timezone.utc)
    md_files = list_md_files(VAULT_PATH)

    reconcile = pass_index_reconcile()
    broken_links = pass_broken_wikilinks(VAULT_PATH, md_files)
    archive_candidates = pass_archive_candidates(VAULT_PATH, md_files, now)
    hot_md_flags = pass_hot_md_budget(VAULT_PATH, md_files)
    near_dups = pass_near_duplicates(VAULT_PATH, md_files)
    contradiction = pass_contradiction_lint_sunday(VAULT_PATH, VAULT_NAME, now)

    report = build_report(
        VAULT_NAME, now, reconcile, broken_links, archive_candidates, hot_md_flags, near_dups, contradiction
    )

    out_path = report_path_for(VAULT_PATH, VAULT_NAME, now)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(
        f"[dreaming] {VAULT_NAME} {now.strftime('%Y-%m-%d %H:%M')} UTC — "
        f"{len(md_files)} files scanned, {len(broken_links)} broken links, "
        f"{len(archive_candidates)} archive candidates → {out_path}",
        flush=True,
    )
    return out_path


if __name__ == "__main__":
    run()
