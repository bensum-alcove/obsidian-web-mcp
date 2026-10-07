#!/usr/bin/env python3
"""log-rollover.py -- roll append-only vault logs over, per Write Rule 21.

Build: brain-hygiene-v1 (BH-3). Ships in DRY-RUN mode: with no flags it only
prints what it would do. ``--apply`` is a deliberate, separate operator step.

What a roll does (the same steps done by hand on 2026-10-07):
  1. Read the current log and its revision.
  2. Copy it byte-for-byte to ``<name>-<first-date>-to-<last-date>.md``.
  3. Swap the live path to a fresh volume (``continued_from``, the same
     preamble and ``read_policy``, a first entry recording the move) in one
     atomic, revision-guarded replace.
  The live path is never missing: an append during a roll either lands in the
  old volume before the swap (the swap is refused and the roll retried) or in
  the new volume after it. A failed roll leaves the original untouched.

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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obsidian_vault_mcp import bounded_files, config, vault  # noqa: E402

# The vault and its crons run on Brisbane time (no daylight saving, so UTC+10 is a safe fallback).
try:
    VAULT_TZ = ZoneInfo("Australia/Brisbane")
except ZoneInfoNotFoundError:  # pragma: no cover - minimal container without tzdata
    VAULT_TZ = timezone(timedelta(hours=10), "AEST")


def vault_today(now: datetime | None = None) -> date:
    """Today's date in the vault's timezone (a naive ``now`` is taken as UTC)."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(VAULT_TZ).date()


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
    d = datetime.fromtimestamp(mtime, tz=VAULT_TZ).date()
    return d, d


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


FIRST_ENTRY_RE = re.compile(r"^##\s+\d{4}-\d{2}-\d{2}\b", re.MULTILINE)
EARLIER_RE = re.compile(r"Earlier volumes?:\s*((?:\[\[[^\]]+\]\][,\s]*)+)\.?")
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
MAX_PREAMBLE_CHARS = 2000
MAX_CHAIN_DEPTH = 50

# Writer note for a log whose old volume carried none (the 2026-09-08 changelog
# volume had no note; the 2026-10-07 hand roll added this one).
DEFAULT_WRITER_NOTES = {
    "infrastructure-changelog.md": (
        "> Reverse-chronological: newest entry directly below this note. Insert with `vault_str_replace` "
        "anchored on the current newest heading; never `vault_write` or `vault_append` this file "
        "(Write Rule 13). Rolled over monthly or at 500KB, whichever comes first (Write Rule 21)."
    ),
}


def _link_target(rel: str) -> str:
    return rel[:-3] if rel.endswith(".md") else rel


def _preamble(body: str) -> str:
    """The writer note between the H1 and the first dated entry; '' if the volume has none."""
    h1 = H1_RE.search(body)
    first = FIRST_ENTRY_RE.search(body)
    if not h1 or not first or first.start() < h1.end():
        return ""
    return body[h1.end():first.start()].strip()[:MAX_PREAMBLE_CHARS]


def _earlier_volumes(rolled_rel: str, text: str, meta: dict, preamble: str) -> list[str]:
    """Link targets for every earlier volume, newest first: the volume being rolled,
    then those already linked in its preamble, then the rest of its continued_from chain."""
    seen: list[str] = []

    def add(target: str) -> None:
        if target and target not in seen:
            seen.append(target)

    add(_link_target(rolled_rel))
    for m in EARLIER_RE.finditer(preamble):
        for t in WIKILINK_RE.findall(m.group(1)):
            add(t)
    prev = meta.get("continued_from")
    for _ in range(MAX_CHAIN_DEPTH):
        if not isinstance(prev, str) or not prev:
            break
        add(_link_target(prev))
        try:
            prev_text, _md = vault.read_file(prev)
        except Exception:
            break
        prev = bounded_files.split_frontmatter(prev_text)[0].get("continued_from")
    return seen


def _writer_note(rel: str, preamble: str, earlier: list[str]) -> str:
    """The old volume's note (or the log's default), with its 'Earlier volumes' list brought up to date."""
    note = preamble or DEFAULT_WRITER_NOTES.get(Path(rel).name, "")
    links = ", ".join(f"[[{t}]]" for t in earlier)
    if EARLIER_RE.search(note):
        return EARLIER_RE.sub(lambda _m: f"Earlier volumes: {links}.", note, count=1)
    sentence = f"Earlier volumes: {links}."
    if not note:
        return f"> {sentence}"
    return f"{note} {sentence}" if note.lstrip().startswith(">") and "\n" not in note else f"{note}\n> {sentence}"


def _new_volume(rel: str, rolled_rel: str, text: str, meta: dict, size: int, revision: str,
                first: date, last: date, today: date) -> str:
    _, body = bounded_files.split_frontmatter(text)
    h1 = H1_RE.search(body)
    title = h1.group(1) if h1 else Path(rel).stem
    preamble = _preamble(body)
    note = _writer_note(rel, preamble, _earlier_volumes(rolled_rel, text, meta, preamble))
    lines = ["---", f"continued_from: {_quote(rolled_rel)}", f"created: {_quote(today.isoformat())}"]
    if meta.get("read_policy") is not None:
        lines.append(f"read_policy: {_quote(str(meta['read_policy']))}")
    if isinstance(meta.get("type"), str):
        lines.append(f"type: {_quote(meta['type'])}")
    if isinstance(meta.get("tags"), list):
        lines.append(f"tags: {json.dumps([str(t) for t in meta['tags']], ensure_ascii=False)}")
    lines += [
        f"updated: {_quote(today.isoformat())}",
        'last_edited_by: "log-rollover (scripts/log-rollover.py)"',
        f"last_edit_note: {_quote(f'{today.isoformat()}: rollover (Write Rule 21). Previous volume copied byte-for-byte; this file starts empty apart from the rollover entry.')}",
        "---",
        "",
        f"# {title}",
        "",
        note,
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
    """Roll one log without the live path ever being missing or a bare fragment.

    1. Write the old volume's bytes to ``rolled_path`` (a copy; the live file is
       untouched and still takes appends).
    2. Replace the live path with the new volume in one atomic swap guarded by
       ``expected_revision``. ``vault.write_file_atomic`` does the check and the
       replace under the same per-path lock ``vault_append`` takes, so an append
       either landed before the swap (the revision no longer matches, the swap is
       refused and the roll is abandoned with the original intact) or lands after
       it, in the new volume.
    3. If the swap fails for any reason, remove the copy; the original was never
       moved, so there is nothing to restore.
    """
    rel, rolled_rel = plan["path"], plan["rolled_path"]
    meta, _ = bounded_files.split_frontmatter(plan["text"])
    new_text = _new_volume(
        rel, rolled_rel, plan["text"], meta, plan["size"], plan["revision"],
        plan["first"], plan["last"], today,
    )
    vault.write_file_atomic(
        rolled_rel, plan["text"], tool="log-rollover", actor="log-rollover", expected_revision="absent",
    )
    try:
        vault.write_file_atomic(
            rel, new_text, tool="log-rollover", actor="log-rollover", expected_revision=plan["revision"],
        )
    except BaseException:
        try:
            vault.delete_path(rolled_rel, expected_revision=plan["revision"], actor="log-rollover")
        except Exception:
            pass  # the original is intact either way; a stray copy is harmless and reported by the raise
        raise


# A roll abandoned because an append landed mid-roll is retried against the new content.
ROLL_ATTEMPTS = 3


def _apply_with_retry(plan: dict, mode: str, today: date) -> dict | None:
    """Apply a roll; if an append landed mid-roll, re-plan and try again.

    Returns the plan that was applied, or None if the retry found the log no longer due.
    """
    for attempt in range(ROLL_ATTEMPTS):
        try:
            apply_roll(plan, today)
            return plan
        except vault.RevisionConflictError:
            if attempt == ROLL_ATTEMPTS - 1:
                raise
            plan = plan_roll(plan["path"], mode, today)
            if not plan["due"]:
                return None
    return None


def run(mode: str = "if-due", apply: bool = False, today: date | None = None,
        now: datetime | None = None) -> list[dict]:
    today = today or vault_today(now)
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
                plan = _apply_with_retry(plan, mode, today)
                if plan is None:
                    results.append({"path": rel, "action": "not-due", "size": entry["size"]})
                    continue
                entry.update(rolled_path=plan["rolled_path"], size=plan["size"], reasons=plan["reasons"])
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
