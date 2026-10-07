#!/usr/bin/env python3
"""log-rollover.py -- roll append-only vault logs over, per Write Rule 21.

Build: brain-hygiene-v1 (BH-3). Ships in DRY-RUN mode: with no flags it only
prints what it would do. ``--apply`` is a deliberate, separate operator step.

What a roll does (the same steps done by hand on 2026-10-07):
  1. Read the current log and its revision.
  2. Move it byte-for-byte to ``<name>-<first-date>-to-<last-date>.md`` with
     ``expected_revision`` (so a write landing mid-roll aborts the roll).
  3. Create a fresh volume at the original path carrying ``continued_from``,
     the same ``read_policy``, and a first entry recording the move.
  If step 3 fails the moved volume is moved back, so the log is never left
  missing.

Which logs, and when:
  --mode if-due   (default) roll logs at or above the rollover size
                  (obsidian_vault_mcp.bounded_files.LOG_ROLLOVER_BYTES).
                  Meant for the same cron that runs dreaming.
  --mode monthly  also roll logs whose volume holds entries from before the
                  current month. Meant for the 1st of the month.

Never touches hot.md (those stay a human/agent rewrite). Only the logs listed
in bounded_files.APPEND_ONLY_LOGS are ever candidates.

Before turning on --apply for a log that a skill, job or build reads by path,
check those readers (Write Rule 21): the live path keeps working, but anything
that expects the old entries at that path will now find them in the volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obsidian_vault_mcp import bounded_files, config, vault  # noqa: E402

DATE_HEADING_RE = re.compile(r"^#{1,3}\s+(\d{4}-\d{2}-\d{2})\b", re.MULTILINE)
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def _entry_dates(text: str) -> list[date]:
    out = []
    for m in DATE_HEADING_RE.finditer(text):
        try:
            out.append(date.fromisoformat(m.group(1)))
        except ValueError:
            continue
    return out


def _volume_span(text: str, mtime: float) -> tuple[date, date]:
    """First and last entry dates of a volume; falls back to frontmatter, then mtime."""
    dates = _entry_dates(text)
    if dates:
        return min(dates), max(dates)
    meta, _ = bounded_files.split_frontmatter(text)
    fallback = []
    for key in ("created", "updated"):
        raw = meta.get(key)
        if isinstance(raw, datetime):
            fallback.append(raw.date())
        elif isinstance(raw, date):
            fallback.append(raw)
        elif isinstance(raw, str):
            try:
                fallback.append(date.fromisoformat(raw[:10]))
            except ValueError:
                pass
    if fallback:
        return min(fallback), max(fallback)
    d = datetime.fromtimestamp(mtime, tz=timezone.utc).date()
    return d, d


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _new_volume(rel: str, rolled_rel: str, text: str, meta: dict, size: int, revision: str,
                first: date, last: date, today: date) -> str:
    _, body = bounded_files.split_frontmatter(text)
    h1 = H1_RE.search(body)
    title = h1.group(1) if h1 else Path(rel).stem
    lines = ["---"]
    if isinstance(meta.get("type"), str):
        lines.append(f"type: {_quote(meta['type'])}")
    if isinstance(meta.get("tags"), list):
        lines.append(f"tags: {json.dumps([str(t) for t in meta['tags']], ensure_ascii=False)}")
    if meta.get("read_policy") is not None:
        lines.append(f"read_policy: {_quote(str(meta['read_policy']))}")
    lines += [
        f"continued_from: {_quote(rolled_rel)}",
        f"created: {_quote(today.isoformat())}",
        f"updated: {_quote(today.isoformat())}",
        "---",
        f"# {title}",
        "",
        f"Earlier entries: [[{rolled_rel[:-3] if rolled_rel.endswith('.md') else rolled_rel}]].",
        "",
        f"## {today.isoformat()} — Log rolled over",
        f"The previous volume ({first.isoformat()} to {last.isoformat()}, {size} bytes, "
        f"revision `{revision}`) was moved byte-for-byte to `{rolled_rel}` by "
        "`scripts/log-rollover.py` under Write Rule 21. New entries start here.",
        "",
    ]
    return "\n".join(lines)


def plan_roll(rel: str, mode: str, today: date) -> dict:
    """Decide whether one log rolls. Reads only; never writes."""
    text, md = vault.read_file(rel)
    size = md["size"]
    first, last = _volume_span(text, Path(config.VAULT_PATH / rel).stat().st_mtime)
    reasons = []
    if size >= bounded_files.LOG_ROLLOVER_BYTES:
        reasons.append(f"size {size} >= {bounded_files.LOG_ROLLOVER_BYTES}")
    if mode == "monthly" and (first.year, first.month) < (today.year, today.month):
        reasons.append(f"volume starts {first.isoformat()}, before {today.strftime('%Y-%m')}")
    rolled_rel = str(Path(rel).with_name(f"{Path(rel).stem}-{first.isoformat()}-to-{last.isoformat()}.md"))
    return {
        "path": rel, "due": bool(reasons), "reasons": reasons, "size": size,
        "revision": md["revision"], "first": first, "last": last, "rolled_path": rolled_rel,
        "text": text,
    }


def apply_roll(plan: dict, today: date) -> None:
    rel, rolled_rel = plan["path"], plan["rolled_path"]
    meta, _ = bounded_files.split_frontmatter(plan["text"])
    new_text = _new_volume(
        rel, rolled_rel, plan["text"], meta, plan["size"], plan["revision"],
        plan["first"], plan["last"], today,
    )
    vault.move_path(rel, rolled_rel, expected_revision=plan["revision"], actor="log-rollover")
    try:
        vault.write_file_atomic(rel, new_text, tool="log-rollover", actor="log-rollover", expected_revision="absent")
    except Exception:
        # Put the log back rather than leave the live path empty.
        vault.move_path(rolled_rel, rel, expected_revision=plan["revision"], actor="log-rollover")
        raise


def run(mode: str = "if-due", apply: bool = False, today: date | None = None) -> list[dict]:
    today = today or datetime.now(timezone.utc).date()
    results = []
    for rel in bounded_files.existing_logs(config.VAULT_PATH):
        if Path(rel).name.lower() == "hot.md":  # defensive: hot.md is never rolled
            continue
        try:
            plan = plan_roll(rel, mode, today)
        except Exception as e:
            results.append({"path": rel, "action": "error", "detail": str(e)})
            continue
        if not plan["due"]:
            results.append({"path": rel, "action": "not-due", "size": plan["size"]})
            continue
        entry = {
            "path": rel, "rolled_path": plan["rolled_path"], "size": plan["size"],
            "reasons": plan["reasons"],
        }
        if not apply:
            entry["action"] = "dry-run"
        else:
            try:
                apply_roll(plan, today)
                entry["action"] = "rolled"
                entry["sha256"] = hashlib.sha256(
                    (config.VAULT_PATH / plan["rolled_path"]).read_bytes()
                ).hexdigest()
            except Exception as e:
                entry["action"] = "error"
                entry["detail"] = str(e)
        results.append(entry)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["if-due", "monthly"], default="if-due")
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually roll the logs. Without it this is a dry run that only prints the plan.",
    )
    args = parser.parse_args(argv)

    label = "APPLY" if args.apply else "DRY-RUN"
    results = run(mode=args.mode, apply=args.apply)
    print(f"[log-rollover] {label} vault={config.VAULT_PATH.name} mode={args.mode}", flush=True)
    for r in results:
        if r["action"] in ("dry-run", "rolled"):
            verb = "would roll" if r["action"] == "dry-run" else "rolled"
            print(f"[log-rollover] {verb} {r['path']} -> {r['rolled_path']} ({'; '.join(r['reasons'])})", flush=True)
        elif r["action"] == "error":
            print(f"[log-rollover] ERROR {r['path']}: {r['detail']}", file=sys.stderr, flush=True)
        else:
            print(f"[log-rollover] not due: {r['path']} ({r['size']} bytes)", flush=True)
    return 1 if any(r["action"] == "error" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
