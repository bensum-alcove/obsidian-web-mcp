#!/usr/bin/env python3
"""hot-md-curate.py — deterministic hot.md budget enforcement and stale-claim detection.

Implements Personal/Build Orchestrator/specs/hot-md-curation.md (build_id: hot-md-curation).
No LLM calls, no API key, no local model — every check here is string/file work.

Check B build-log search note: the spec's literal instruction only looks in
`Personal/Build Orchestrator/build-logs/`. In practice several referenced build
outputs (fix-bo-spec-path-prefix, bo-auto-sync-ingest, bo-codex-engine-routing)
live directly under `BS 2nd Brain/Alcove/Infrastructure/` instead. The spec's own
Step 4 says "if Check B does not flag fix-bo-spec-path-prefix, the check is wrong,
fix it" — so the build-log search below checks both locations.

Status parsing note: none of the three real build logs above use a YAML
frontmatter `status:` field — status shows up as body text ("## Status\npass",
"## Status: pass", "**Status: pass**"). get_status() strips markdown emphasis
characters and matches "status" followed by a status word, so it catches all
three forms plus real frontmatter status fields.

Check E (build_id: hot-md-curate-scheduled-reviews): additive-only. Parses
scheduled-reviews.md for pending reviews due today or earlier and surfaces
them in the report and Telegram summary. Report-only — never writes anything,
in either mode. Missing/unparseable file degrades to a single warning line;
Checks A-D still run and the script still exits 0.

v2 additions (build_id: hot-md-curate-v2): --apply now also rotates Check B
RESOLVED/STALE-BLOCKER bullets to hot-archive/ (previously report-only), and
runs a char-budget enforcement pass afterward that rotates the oldest bullet
from "Last session shipped" then "In flight" (floor 2 each) until the file is
under budget or the floor is hit (OVER-BUDGET-AT-FLOOR). Bullets over 250
chars are flagged as LONG-BULLET (report-only, never auto-edited). This build
also found BUILD_LOG_SEARCH_DIRS missing the "Infrastructure/Build Logs/"
subdirectory where several real build logs live — added below.
"""

import argparse
import fcntl
import os
import re
import shutil
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime

VAULT_ROOT_DEFAULT = "/home/ben_sum/vaults/bs-brain"

TARGETS = [
    "BS 2nd Brain/Alcove/Infrastructure/hot.md",
    "BS 2nd Brain/Alcove/Skills/hot.md",
    "Personal/Build Orchestrator/hot.md",
]

BUDGET_CHARS = 2500
SECTION_CAP = 5
BUDGET_FLOOR = 2
LONG_BULLET_CHARS = 250
MTIME_GUARD_SECONDS = 15 * 60
ARCHIVE_DIR = "hot-archive"

BUILD_LOG_SEARCH_DIRS = [
    "Personal/Build Orchestrator/build-logs",
    "BS 2nd Brain/Alcove/Infrastructure/Build Logs",
    "BS 2nd Brain/Alcove/Infrastructure",
]
SPEC_DIR = "Personal/Build Orchestrator/specs"

SCHEDULED_REVIEWS_PATH = "BS 2nd Brain/Alcove/Infrastructure/scheduled-reviews.md"

STALE_BLOCKER_KEYWORDS = [
    "blocking",
    "blocked",
    "pending",
    "awaiting",
    "not yet",
    "do not change",
]
RESOLVED_STATUS_WORDS = {"pass", "passed", "complete", "completed"}

REVIEW_HEADING_RE = re.compile(r"^###\s+(\S+)\s*$")
REVIEW_DUE_RE = re.compile(r"^due:\s*(\d{4}-\d{2}-\d{2})\s*$", re.IGNORECASE)
REVIEW_STATUS_RE = re.compile(r"^status:\s*(\S+)\s*$", re.IGNORECASE)

BACKUP_ROOT = os.path.expanduser("~/backups/hot-md-curate")
LOCK_FILE = "/tmp/hot-md-curate.lock"

TELEGRAM_BOT_TOKEN = "8954494669:AAEQdrVGbRRmowTI1TZTsCUD5VBFmJL9dlM"
TELEGRAM_CHAT_ID = "8558481275"

FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)---\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,2})\s+(.*?)\s*$")
BACKTICK_CANDIDATE_RE = re.compile(r"`([a-z0-9]+(?:-[a-z0-9]+){1,6})`")
BARE_CANDIDATE_RE = re.compile(r"\b([a-z0-9]+(?:-[a-z0-9]+){1,6})\b")


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def frontmatter_and_body(text):
    m = FRONTMATTER_RE.match(text)
    if not m:
        return "", text
    return m.group(0), text[m.end():]


def char_count_excluding_frontmatter(text):
    _, body = frontmatter_and_body(text)
    return len(body)


def bump_frontmatter_updated(text, date_str):
    fm, body = frontmatter_and_body(text)
    if not fm:
        return text
    new_fm, n = re.subn(
        r"(?m)^(updated:\s*)('?)[0-9]{4}-[0-9]{2}-[0-9]{2}\2\s*$",
        lambda m: f"{m.group(1)}'{date_str}'",
        fm,
    )
    if n == 0:
        return text
    return new_fm + body


def line_candidates(line):
    found = set()
    for tok in BACKTICK_CANDIDATE_RE.findall(line):
        found.add(tok)
    for tok in BARE_CANDIDATE_RE.findall(line):
        found.add(tok)
    return found


def get_status(build_log_text):
    clean = re.sub(r"[#*`'\"]", "", build_log_text)
    m = re.search(r"\bstatus\s*:?\s*([a-zA-Z]+)", clean, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower()


def find_spec_path(vault_root, candidate):
    p = os.path.join(vault_root, SPEC_DIR, f"{candidate}.md")
    return p if os.path.isfile(p) else None


def find_build_log_path(vault_root, candidate):
    for d in BUILD_LOG_SEARCH_DIRS:
        p = os.path.join(vault_root, d, f"{candidate}-output.md")
        if os.path.isfile(p):
            return p
    return None


def check_a(text):
    chars = char_count_excluding_frontmatter(text)
    return {
        "chars": chars,
        "budget": BUDGET_CHARS,
        "overage": max(0, chars - BUDGET_CHARS),
    }


def check_b(vault_root, all_lines, scan_start, sections):
    """Returns (report_lines, removal_specs).

    removal_specs is a list of (line_index_set, reason_prefix, block_text) for
    bullet blocks that should be rotated verbatim to hot-archive/ in apply
    mode. scan_start is the first index into all_lines that is outside the
    YAML frontmatter, so frontmatter fields (e.g. build_id:) are never treated
    as candidates.
    """
    candidate_line_idxs = {}
    for i in range(scan_start, len(all_lines)):
        for cand in line_candidates(all_lines[i]):
            candidate_line_idxs.setdefault(cand, []).append(i)

    blocks = []
    for sec in sections:
        if sec["level"] != 2:
            continue
        sec_blocks, _ = parse_bullet_blocks(all_lines, sec["start"], sec["end"])
        blocks.extend(sec_blocks)

    def block_for_line(i):
        for (s, e) in blocks:
            if s <= i < e:
                return (s, e)
        return None

    report = []
    removal_specs = []
    added_blocks = set()
    for candidate in sorted(candidate_line_idxs):
        spec_path = find_spec_path(vault_root, candidate)
        if not spec_path:
            continue
        log_path = find_build_log_path(vault_root, candidate)
        if not log_path:
            continue
        status = get_status(read_file(log_path))
        if status not in RESOLVED_STATUS_WORDS:
            continue

        rel_log = os.path.relpath(log_path, vault_root)
        idxs = candidate_line_idxs[candidate]
        severity = "RESOLVED"
        for i in idxs:
            low = all_lines[i].lower()
            if any(k in low for k in STALE_BLOCKER_KEYWORDS):
                severity = "STALE-BLOCKER"
                break

        report.append(
            f"{severity}: {candidate} — build log shows status: {status} "
            f"({rel_log}), but hot.md still references it"
        )

        reason = "resolved" if severity == "RESOLVED" else "stale-blocker"
        for i in idxs:
            blk = block_for_line(i)
            if blk is None or blk in added_blocks:
                continue
            added_blocks.add(blk)
            s, e = blk
            removal_specs.append(
                (frozenset(range(s, e)), f"[{reason}: {candidate}]", "\n".join(all_lines[s:e]))
            )

    return report, removal_specs


def parse_sections(lines):
    boundaries = [i for i, l in enumerate(lines) if HEADING_RE.match(l)]
    sections = []
    for idx, i in enumerate(boundaries):
        m = HEADING_RE.match(lines[i])
        level = len(m.group(1))
        heading_text = m.group(2)
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
        sections.append(
            {"level": level, "heading": heading_text, "start": i + 1, "end": end}
        )
    return sections


def parse_bullet_blocks(lines, start, end):
    blocks = []
    loose = []
    i = start
    while i < end:
        if re.match(r"^-\s", lines[i]):
            j = i + 1
            while j < end and lines[j] != "" and lines[j][0] in " \t":
                j += 1
            blocks.append((i, j))
            i = j
        else:
            loose.append(i)
            i += 1
    return blocks, loose


def is_capped_section(heading):
    h = heading.strip().lower()
    return h.startswith("last session shipped") or h.startswith("in flight")


def is_watchpoint_section(heading):
    h = heading.strip().lower()
    return "blockers" in h or "watchpoints" in h


def check_c(lines, sections, rel_path):
    """Returns (report_lines, remove_line_indices, archive_chunks)."""
    report = []
    remove = set()
    archive_chunks = []
    for sec in sections:
        if sec["level"] != 2 or not is_capped_section(sec["heading"]):
            continue
        blocks, _ = parse_bullet_blocks(lines, sec["start"], sec["end"])
        if len(blocks) <= SECTION_CAP:
            report.append(
                f'Section "{sec["heading"]}": {len(blocks)} bullets (cap {SECTION_CAP}) — within budget'
            )
            continue
        overflow_count = len(blocks) - SECTION_CAP
        overflow_blocks = blocks[:overflow_count]
        report.append(
            f'Section "{sec["heading"]}": {len(blocks)} bullets (cap {SECTION_CAP}) — '
            f"{overflow_count} to rotate to {ARCHIVE_DIR}/"
        )
        for (s, e) in overflow_blocks:
            remove.update(range(s, e))
            archive_chunks.append("\n".join(lines[s:e]))
    return report, remove, archive_chunks


def check_d(lines, sections):
    """Returns (report_lines, remove_line_indices)."""
    report = []
    remove = set()
    flag_re = re.compile(r"~~|removed 2026-|resolved|superseded", re.IGNORECASE)
    for sec in sections:
        if sec["level"] != 2 or not is_watchpoint_section(sec["heading"]):
            continue
        blocks, _ = parse_bullet_blocks(lines, sec["start"], sec["end"])
        for (s, e) in blocks:
            text = "\n".join(lines[s:e])
            if flag_re.search(text):
                report.append(f'ROTATE-CANDIDATE: "{lines[s].strip()}"')
                remove.update(range(s, e))
    return report, remove


def render_with_removals(lines, remove, sections):
    touched_ranges = [
        (sec["start"], sec["end"])
        for sec in sections
        if sec["level"] == 2 and any(i in remove for i in range(sec["start"], sec["end"]))
    ]

    def in_touched(i):
        return any(s <= i < e for s, e in touched_ranges)

    new_lines = []
    prev_blank_collapsible = False
    for i, l in enumerate(lines):
        if i in remove:
            continue
        if l == "" and in_touched(i):
            if prev_blank_collapsible:
                continue
            prev_blank_collapsible = True
        else:
            prev_blank_collapsible = False
        new_lines.append(l)
    return new_lines


def long_bullet_check(lines, sections, rel_path):
    report = []
    for sec in sections:
        if sec["level"] != 2 or not is_capped_section(sec["heading"]):
            continue
        blocks, _ = parse_bullet_blocks(lines, sec["start"], sec["end"])
        for (s, e) in blocks:
            text = "\n".join(lines[s:e])
            if len(text) > LONG_BULLET_CHARS:
                snippet = text.strip().replace("\n", " ")[:60]
                report.append(
                    f'LONG-BULLET: {rel_path} — section "{sec["heading"]}" — "{snippet}"'
                )
    return report


def budget_enforcement_pass(lines):
    """Iteratively rotates the oldest bullet from "Last session shipped" then
    "In flight" (never below BUDGET_FLOOR bullets in either) until the text
    is under BUDGET_CHARS. Never touches any other section or frontmatter.
    Returns (lines, archive_entries, over_budget)."""
    archive_entries = []
    section_order = ["last session shipped", "in flight"]

    def current_chars():
        return char_count_excluding_frontmatter("\n".join(lines) + "\n")

    for prefix in section_order:
        while current_chars() > BUDGET_CHARS:
            sections = parse_sections(lines)
            sec = next(
                (s for s in sections if s["level"] == 2 and s["heading"].strip().lower().startswith(prefix)),
                None,
            )
            if sec is None:
                break
            blocks, _ = parse_bullet_blocks(lines, sec["start"], sec["end"])
            if len(blocks) <= BUDGET_FLOOR:
                break
            s, e = blocks[0]
            block_text = "\n".join(lines[s:e])
            archive_entries.append((f"[budget-rotate: {sec['heading'].strip()}]", block_text))
            lines = render_with_removals(lines, set(range(s, e)), sections)
        if current_chars() <= BUDGET_CHARS:
            break

    return lines, archive_entries, current_chars() > BUDGET_CHARS


def check_e(vault_root, today_str):
    """Report-only. Returns (due_lines, warning_lines) — never writes."""
    path = os.path.join(vault_root, SCHEDULED_REVIEWS_PATH)
    if not os.path.isfile(path):
        return [], [f"WARNING: {SCHEDULED_REVIEWS_PATH} not found; Check E skipped"]

    try:
        lines = read_file(path).split("\n")
    except OSError as e:
        return [], [f"WARNING: failed to read {SCHEDULED_REVIEWS_PATH} ({e}); Check E skipped"]

    heading_idxs = [i for i, l in enumerate(lines) if REVIEW_HEADING_RE.match(l)]
    if not heading_idxs:
        return [], [f"WARNING: no ### review blocks found in {SCHEDULED_REVIEWS_PATH}; Check E skipped"]

    due = []
    for idx, i in enumerate(heading_idxs):
        block_id = REVIEW_HEADING_RE.match(lines[i]).group(1)
        end = heading_idxs[idx + 1] if idx + 1 < len(heading_idxs) else len(lines)
        due_date = None
        status = None
        for l in lines[i + 1:end]:
            stripped = l.strip()
            if due_date is None:
                m = REVIEW_DUE_RE.match(stripped)
                if m:
                    due_date = m.group(1)
                    continue
            if status is None:
                m = REVIEW_STATUS_RE.match(stripped)
                if m:
                    status = m.group(1).lower()
        if status == "pending" and due_date and due_date <= today_str:
            due.append(f"REVIEW DUE: {block_id} (due {due_date})")
    return due, []


def backup_targets(vault_root, ts):
    backup_dir = os.path.join(BACKUP_ROOT, ts)
    for rel in TARGETS:
        src = os.path.join(vault_root, rel)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(backup_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return backup_dir


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT_ID, "text": message}).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode()


def process_target(vault_root, rel_path, apply_mode, today_str, now_ts):
    abs_path = os.path.join(vault_root, rel_path)
    result = {
        "rel_path": rel_path,
        "exists": os.path.isfile(abs_path),
        "mtime_skipped": False,
        "changed": False,
        "report_lines": [],
        "resolved_findings": [],
        "rotate_candidates": [],
    }
    if not result["exists"]:
        result["report_lines"].append(f"MISSING: {rel_path} not found")
        return result

    text = read_file(abs_path)
    a = check_a(text)
    result["check_a"] = a
    result["report_lines"].append(
        f"Chars: {a['chars']} / {a['budget']} budget (over by {a['overage']})"
    )

    fm, body = frontmatter_and_body(text)
    fm_lines_count = fm.count("\n")
    all_lines = text.split("\n")
    if all_lines and all_lines[-1] == "":
        all_lines = all_lines[:-1]
        trailing_newline = True
    else:
        trailing_newline = False

    sections = parse_sections(all_lines)

    b_report, b_removal_specs = check_b(vault_root, all_lines, fm_lines_count, sections)
    result["resolved_findings"] = b_report
    result["report_lines"].extend(b_report)

    if apply_mode:
        mtime = os.path.getmtime(abs_path)
        if now_ts - mtime < MTIME_GUARD_SECONDS:
            result["mtime_skipped"] = True
            result["report_lines"].append(
                f"SKIPPED (apply): mtime within last {MTIME_GUARD_SECONDS // 60} minutes "
                "— possible live edit in progress"
            )
            c_report, _, _ = check_c(all_lines, sections, rel_path)
            d_report, _ = check_d(all_lines, sections)
            result["report_lines"].extend(c_report)
            result["rotate_candidates"] = d_report
            result["report_lines"].extend(d_report)
            return result

    c_report, c_remove, c_archive = check_c(all_lines, sections, rel_path)
    d_report, d_remove = check_d(all_lines, sections)
    result["report_lines"].extend(c_report)
    result["rotate_candidates"] = d_report
    result["report_lines"].extend(d_report)

    if not apply_mode:
        result["report_lines"].extend(long_bullet_check(all_lines, sections, rel_path))
        return result

    existing_remove = c_remove | d_remove
    b_remove = set()
    b_archive = []
    for rng, prefix, chunk in b_removal_specs:
        if rng & existing_remove:
            continue  # already being rotated by Check C/D — avoid double-archiving
        b_remove |= rng
        b_archive.append((prefix, chunk))

    remove = existing_remove | b_remove
    archive_entries = [(None, chunk) for chunk in c_archive] + b_archive

    working_lines = render_with_removals(all_lines, remove, sections) if remove else list(all_lines)
    working_lines, budget_archive, over_budget = budget_enforcement_pass(working_lines)
    archive_entries.extend(budget_archive)

    result["report_lines"].extend(
        long_bullet_check(working_lines, parse_sections(working_lines), rel_path)
    )
    if over_budget:
        final_chars = char_count_excluding_frontmatter(
            "\n".join(working_lines) + ("\n" if trailing_newline else "")
        )
        result["report_lines"].append(f"OVER-BUDGET-AT-FLOOR: {rel_path} ({final_chars} chars)")

    if working_lines == all_lines:
        return result

    new_text = "\n".join(working_lines) + ("\n" if trailing_newline else "")
    new_text = bump_frontmatter_updated(new_text, today_str)

    write_file(abs_path, new_text)
    result["changed"] = True

    if archive_entries:
        archive_path = os.path.join(vault_root, ARCHIVE_DIR, f"{today_str[:7]}.md")
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        header = f"## Rotated {today_str} from {rel_path}\n\n"
        pieces = [
            f"{prefix} {today_str}\n{chunk}" if prefix else chunk
            for prefix, chunk in archive_entries
        ]
        chunk_text = header + "\n".join(pieces) + "\n\n"
        with open(archive_path, "a", encoding="utf-8") as f:
            f.write(chunk_text)
        result["report_lines"].append(
            f"Archived {len(archive_entries)} bullet block(s) to {os.path.relpath(archive_path, vault_root)}"
        )

    if d_remove:
        result["report_lines"].append(
            f"Removed {len(d_report)} resolved watchpoint bullet(s) in place (not archived)"
        )

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Perform Check C/D writes")
    parser.add_argument("--report", action="store_true", help="Report only (default)")
    parser.add_argument("--vault-root", default=VAULT_ROOT_DEFAULT)
    parser.add_argument("--no-telegram", action="store_true")
    parser.add_argument(
        "--from-cron", action="store_true",
        help="Set by the cron line only. Manual/build-invoked runs never notify Telegram "
             "(bo-awaiting-input-precision) — the dated report file is the audit trail for those.",
    )
    args = parser.parse_args()

    apply_mode = bool(args.apply)

    lock_fp = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another hot-md-curate.py instance is running; exiting.")
        sys.exit(0)

    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    now_ts = time.time()

    if apply_mode:
        ts = now.strftime("%Y-%m-%dT%H-%M-%SZ")
        backup_dir = backup_targets(args.vault_root, ts)
    else:
        backup_dir = None

    all_results = []
    for rel in TARGETS:
        res = process_target(args.vault_root, rel, apply_mode, today_str, now_ts)
        all_results.append(res)

    report_lines = []
    report_lines.append(f"# Hot.md Curation Report — {today_str}")
    report_lines.append("")
    report_lines.append(f"Mode: {'apply' if apply_mode else 'report'}")
    if backup_dir:
        report_lines.append(f"Backup: {backup_dir}")
    report_lines.append("")

    total_resolved = 0
    total_stale = 0
    total_rotate = 0
    tg_lines = []

    for res in all_results:
        report_lines.append(f"## {res['rel_path']}")
        if not res["exists"]:
            report_lines.append("- MISSING")
            report_lines.append("")
            continue
        for l in res["report_lines"]:
            report_lines.append(f"- {l}")
        report_lines.append("")

        resolved = sum(1 for f in res["resolved_findings"] if f.startswith("RESOLVED"))
        stale = sum(1 for f in res["resolved_findings"] if f.startswith("STALE-BLOCKER"))
        rotate = len(res["rotate_candidates"])
        total_resolved += resolved
        total_stale += stale
        total_rotate += rotate

        a = res.get("check_a", {})
        tg_lines.append(
            f"{res['rel_path']}: {a.get('chars', '?')}/{BUDGET_CHARS} chars "
            f"(RESOLVED={resolved} STALE-BLOCKER={stale} ROTATE-CANDIDATE={rotate})"
        )

    review_due, review_warnings = check_e(args.vault_root, today_str)
    report_lines.append("## Scheduled Reviews")
    for l in review_due:
        report_lines.append(f"- {l}")
    for l in review_warnings:
        report_lines.append(f"- {l}")
    if not review_due and not review_warnings:
        report_lines.append("- None due")
    report_lines.append("")

    report_text = "\n".join(report_lines) + "\n"

    report_path = os.path.join(
        args.vault_root,
        "BS 2nd Brain/Alcove/Infrastructure/hot-md-reports",
        f"{today_str}.md",
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    write_file(report_path, report_text)

    print(report_text)
    print(f"Report written to {report_path}")

    # Notifications are quiet unless the run is both cron-invoked and actionable
    # (bo-awaiting-input-precision Part B). A clean run's dated report file above
    # is the audit trail; Telegram is reserved for things needing attention.
    invoked_by_cron = args.from_cron or os.environ.get("HOT_MD_CURATE_FROM_CRON") == "1"
    actionable = (
        any(res["changed"] for res in all_results)
        or total_resolved > 0
        or total_stale > 0
        or len(review_due) > 0
        or "OVER-BUDGET-AT-FLOOR" in report_text
        or "LONG-BULLET" in report_text
        or "CONTENT-STALE" in report_text
    )

    if args.no_telegram:
        pass
    elif not invoked_by_cron:
        print("Telegram notify skipped: not invoked by cron (manual/build-invoked run)")
    elif not actionable:
        print("Telegram notify skipped: no actionable findings (clean run)")
    else:
        tg_message = (
            f"hot-md-curate ({'apply' if apply_mode else 'report'}) — {today_str}\n"
            + "\n".join(tg_lines)
            + f"\nTotals: RESOLVED={total_resolved} STALE-BLOCKER={total_stale} "
            f"ROTATE-CANDIDATE={total_rotate} REVIEW-DUE={len(review_due)}"
        )
        if review_due:
            tg_message += "\n" + "\n".join(review_due)
        if review_warnings:
            tg_message += "\n" + "\n".join(review_warnings)
        try:
            resp = send_telegram(tg_message)
            print(f"Telegram response: {resp}")
        except Exception as e:
            print(f"Telegram send failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
