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
them in the dated report. Report-only — never writes anything,
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

v3 additions (build_id: hot-md-curate-v3): budget_enforcement_pass now rotates
from every canonical section (CANONICAL_SECTIONS), not just the two "capped"
ones, so bulk in "What's current"/"Parked"/"Blockers / watchpoints" is no
longer immune — same floor of 2 bullets each, no section ever emptied.
Budget raised 2,500 -> 5,000 chars (Ben's steer 2026-08-14). Any level-2
heading outside CANONICAL_SECTIONS is now named in the report as
NON-CANONICAL-SECTION with its char count — report-only, never rotated or
edited, since deleting arbitrary prose is out of scope. CONTENT-STALE detects
when the newest date appearing anywhere in a file's body is more than
CONTENT_STALE_DAYS old, which is the freshness-lie failure mode this build
exists to catch. The curator no longer bumps frontmatter `updated:` on a
rotation-only run — previously every --apply rotation stamped `updated:` to
today even though rotating stale content out is not a content update, which
is exactly how BO's hot.md ended up claiming 2026-08-14 while its body was
six weeks stale.

v4 additions (build_id: hot-cache-v4): BUDGET_CHARS is now read from
obsidian_vault_mcp.config.HOT_MD_BUDGET_CHARS instead of a hardcoded literal
here -- that constant is the single source of truth also imported by
scripts/dreaming.py's nightly budget flag, which previously carried its own
stale 2,500 copy left over from before the v3 raise and would misreport any
hot.md between 2,501 and 5,000 chars as over-budget. New Check F
(check_f) cross-references component_id-shaped tokens in hot.md against
canonical-state records (obsidian_vault_mcp.canonical_state, read-only): if a
referenced component's current record carries a state in
CANONICAL_CONFLICT_STATES and the hot.md bullet doesn't itself say so, that
bullet is flagged CANONICAL-CONFLICT and rotated to hot-archive/ in apply
mode -- same as every other rotation here, canonical-state is never written
by this tool, and the conflicting hot.md claim always loses, never the other
way around.

v5 additions (build_id: hot-md-curate-exception-only-selfheal-v2): cron apply
holds the existing singleton lock while waiting out recent mtimes, then
re-reads every target and retries within a fixed bound. The dated report keeps
all attempt diagnostics, while a fresh final read is the sole notification
authority. Routine maintenance and REVIEW-DUE bookkeeping are Telegram-silent;
only unresolved write, preservation, parse/structure, or retry-exhaustion
exceptions can produce one concise cron notification. Archive bytes are
verified before source removal, including resolved watchpoints that older
versions removed without archiving.
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
from pathlib import Path

_DEV_SRC = "/home/ben_sum/obsidian-web-mcp/src"
if _DEV_SRC not in sys.path:
    sys.path.insert(0, _DEV_SRC)
from obsidian_vault_mcp import canonical_state, config, vault_lock  # noqa: E402

VAULT_ROOT_DEFAULT = "/home/ben_sum/vaults/bs-brain"

TARGETS = [
    "BS 2nd Brain/Alcove/Infrastructure/hot.md",
    "BS 2nd Brain/Alcove/Skills/hot.md",
    "Personal/Build Orchestrator/hot.md",
]

BUDGET_CHARS = config.HOT_MD_BUDGET_CHARS  # single source of truth — see docstring
SECTION_CAP = 5
BUDGET_FLOOR = 2
LONG_BULLET_CHARS = 250
MTIME_GUARD_SECONDS = 15 * 60
ARCHIVE_DIR = "hot-archive"
CONTENT_STALE_DAYS = 14
CRON_RETRY_MARGIN_SECONDS = 5
# Long enough to absorb a fresh edit during the first quiet-window wait, but
# finite so a continually edited target cannot turn the daily cron into a
# daemon. The process-wide LOCK_FILE remains held throughout the wait.
CRON_RETRY_LIMIT_SECONDS = (2 * MTIME_GUARD_SECONDS) + (5 * 60)

# Read-only cross-reference target for Check F — canonical_state.py never writes here.
CANONICAL_STATE_RECORDS_DIR = "BS 2nd Brain/Alcove/Infrastructure/Canonical State/records"
CANONICAL_CONFLICT_STATES = {"deprecated", "broken", "retired", "removed", "superseded"}

# Canonical hot.md structure — see BS 2nd Brain/Alcove/Infrastructure/hot-md-structure.md.
# Order matches document order and is the rotation order in budget_enforcement_pass.
CANONICAL_SECTIONS = [
    "what's current",
    "last session shipped",
    "in flight",
    "parked",
    "blockers / watchpoints",
]

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

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)---\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,2})\s+(.*?)\s*$")
BACKTICK_CANDIDATE_RE = re.compile(r"`([a-z0-9]+(?:-[a-z0-9]+){1,6})`")
BARE_CANDIDATE_RE = re.compile(r"\b([a-z0-9]+(?:-[a-z0-9]+){1,6})\b")
BODY_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, text):
    # Shared cross-process mutation authority (vault-integrity-and-bo-
    # authority-remediation-v2) -- the same lock/atomic-replace primitive
    # the live Vault MCP server's own write_file_atomic uses, so a
    # concurrent MCP write to the same resolved path can no longer be
    # silently overwritten by this --apply run (or vice versa).
    vault_lock.atomic_write(Path(path).resolve(), text.encode("utf-8"))


def frontmatter_and_body(text):
    m = FRONTMATTER_RE.match(text)
    if not m:
        return "", text
    return m.group(0), text[m.end():]


def char_count_excluding_frontmatter(text):
    _, body = frontmatter_and_body(text)
    return len(body)


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


def load_current_canonical_states(records_dir):
    """component_id -> current CanonicalStateRecord, for records with no
    superseded_by. A component_id with more than one current record is a
    duplicate-authority bug that canonical_state_scan.py exists to report —
    this function silently excludes it rather than guessing a winner, since
    picking one would be exactly the kind of authority collision Check F
    must not create."""
    records, _errors = canonical_state.load_all_records(records_dir)
    by_id = {}
    duplicated = set()
    for r in records:
        if not r.is_current:
            continue
        if r.component_id in by_id:
            duplicated.add(r.component_id)
        else:
            by_id[r.component_id] = r
    for cid in duplicated:
        by_id.pop(cid, None)
    return by_id


def check_f(vault_root, all_lines, scan_start, sections):
    """Returns (report_lines, removal_specs) — same shape as check_b.

    Cross-references component_id-shaped tokens in hot.md against
    canonical-state records (read-only; canonical_state.py records are never
    written by this tool). If a referenced component's *current* record
    carries a state in CANONICAL_CONFLICT_STATES and the hot.md bullet
    doesn't itself say so, hot.md is asserting something canonical state has
    already superseded — flag it and rotate the bullet to archive in apply
    mode. On conflict hot.md always loses; canonical state is never rewritten
    to match hot.md.
    """
    records_dir = os.path.join(vault_root, CANONICAL_STATE_RECORDS_DIR)
    current_by_id = load_current_canonical_states(records_dir)
    if not current_by_id:
        return [], []

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
        record = current_by_id.get(candidate)
        if record is None:
            continue
        state = record.state.strip().lower()
        if state not in CANONICAL_CONFLICT_STATES:
            continue
        for i in candidate_line_idxs[candidate]:
            blk = block_for_line(i)
            if blk is None or blk in added_blocks:
                continue
            block_text = "\n".join(all_lines[blk[0]:blk[1]])
            if state in block_text.lower():
                continue  # hot.md already reflects the canonical state itself
            added_blocks.add(blk)
            report.append(
                f"CANONICAL-CONFLICT: {candidate} — canonical state says "
                f"{state!r} ({record.path}), but hot.md does not reflect it"
            )
            removal_specs.append(
                (frozenset(range(blk[0], blk[1])), f"[canonical-conflict: {candidate} -> {state}]", block_text)
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


def is_canonical_section(heading):
    h = heading.strip().lower()
    return any(h.startswith(prefix) for prefix in CANONICAL_SECTIONS)


def non_canonical_section_report(lines, sections, rel_path):
    """Report-only: names any level-2 heading outside CANONICAL_SECTIONS with
    its char count. Never deletes, merges, or edits — that is a human call."""
    report = []
    for sec in sections:
        if sec["level"] != 2 or is_canonical_section(sec["heading"]):
            continue
        text = "\n".join(lines[sec["start"]:sec["end"]])
        report.append(
            f'NON-CANONICAL-SECTION: {rel_path} — "{sec["heading"]}" ({len(text)} chars)'
        )
    return report


def structure_safety_errors(text, sections, rel_path):
    """Return only ambiguities that make deterministic mutation unsafe.

    Ordinary non-canonical sections remain diagnostics. An unterminated
    frontmatter block or an exactly duplicated canonical heading is different:
    continuing would make line ownership ambiguous, so apply mode must leave
    the target untouched and surface one integrity exception. Distinct allowed
    suffix variants (for example two differently-qualified Parked sections)
    remain valid.
    """
    errors = []
    if text.startswith("---\n") and FRONTMATTER_RE.match(text) is None:
        errors.append(f"{rel_path}: malformed or unterminated YAML frontmatter")

    canonical_headings = [
        sec["heading"].strip().lower()
        for sec in sections
        if sec["level"] == 2 and is_canonical_section(sec["heading"])
    ]
    for heading in sorted(set(canonical_headings)):
        if canonical_headings.count(heading) > 1:
            errors.append(
                f"{rel_path}: duplicate canonical section {heading!r} blocks safe curation"
            )
    return errors


def content_stale_check(lines, scan_start, today_str, rel_path):
    """Report-only: flags rel_path if the newest YYYY-MM-DD date found anywhere
    in the body (headings or inline) is more than CONTENT_STALE_DAYS before
    today. Lexical max is safe here since all matches are well-formed ISO dates."""
    dates = []
    for l in lines[scan_start:]:
        dates.extend(BODY_DATE_RE.findall(l))
    if not dates:
        return None
    newest = max(dates)
    age_days = (
        datetime.strptime(today_str, "%Y-%m-%d") - datetime.strptime(newest, "%Y-%m-%d")
    ).days
    if age_days > CONTENT_STALE_DAYS:
        return f"CONTENT-STALE: {rel_path} — newest body date {newest} is {age_days} days old (threshold {CONTENT_STALE_DAYS})"
    return None


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
    """Returns (report_lines, remove_line_indices, archive_chunks)."""
    report = []
    remove = set()
    archive_chunks = []
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
                archive_chunks.append((frozenset(range(s, e)), text))
    return report, remove, archive_chunks


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
    """Iteratively rotates the oldest bullet from every canonical section, in
    CANONICAL_SECTIONS order, never below BUDGET_FLOOR bullets in any one,
    until the text is under BUDGET_CHARS. Non-canonical sections are never
    touched — their bulk is counted but not rotatable by this tool.
    Returns (lines, archive_entries, over_budget)."""
    archive_entries = []
    section_order = CANONICAL_SECTIONS

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


def backup_targets(vault_root, ts, phase):
    """File-level revert backup. Called with phase="before" pre-mutation and
    phase="after" post-mutation, so an apply-mode run always leaves both the
    pre-image and the post-image on disk under one timestamped run directory
    — a before-only backup can restore, but can't show what a run actually
    changed without diffing against the live (now-mutated) vault file."""
    backup_dir = os.path.join(BACKUP_ROOT, ts, phase)
    for rel in TARGETS:
        src = os.path.join(vault_root, rel)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(backup_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return backup_dir


def append_archive_losslessly(archive_path, chunk_text):
    """Append one complete preservation record and verify its exact bytes.

    ``atomic_append`` serialises against every cooperating vault writer. The
    post-read accepts a later append by another writer, but requires our exact
    contiguous record to be present before the source hot.md is rewritten.
    """
    archive_path = Path(archive_path).resolve()
    data = chunk_text.encode("utf-8")
    vault_lock.atomic_append(archive_path, data)
    if data not in archive_path.read_bytes():
        raise OSError("lossless archive verification failed")


def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set in environment — cannot send Telegram notification"
        )
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
        "content_stale": None,
        "canonical_conflicts": [],
        "safety_exceptions": [],
        "operation_failed": False,
    }
    if not result["exists"]:
        error = f"{rel_path}: target is missing"
        result["safety_exceptions"].append(error)
        result["report_lines"].append(f"SAFETY-EXCEPTION: {error}")
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

    structure_errors = structure_safety_errors(text, sections, rel_path)
    if structure_errors:
        result["safety_exceptions"].extend(structure_errors)
        result["report_lines"].extend(
            f"SAFETY-EXCEPTION: {error}" for error in structure_errors
        )
        return result

    result["report_lines"].extend(non_canonical_section_report(all_lines, sections, rel_path))

    content_stale_line = content_stale_check(all_lines, fm_lines_count, today_str, rel_path)
    result["content_stale"] = content_stale_line
    if content_stale_line:
        result["report_lines"].append(content_stale_line)

    b_report, b_removal_specs = check_b(vault_root, all_lines, fm_lines_count, sections)
    result["resolved_findings"] = b_report
    result["report_lines"].extend(b_report)

    f_report, f_removal_specs = check_f(vault_root, all_lines, fm_lines_count, sections)
    result["canonical_conflicts"] = f_report
    result["report_lines"].extend(f_report)

    if apply_mode:
        mtime = os.path.getmtime(abs_path)
        if now_ts - mtime < MTIME_GUARD_SECONDS:
            result["mtime_skipped"] = True
            result["report_lines"].append(
                f"SKIPPED (apply): mtime within last {MTIME_GUARD_SECONDS // 60} minutes "
                "— possible live edit in progress"
            )
            c_report, _, _ = check_c(all_lines, sections, rel_path)
            d_report, _, _ = check_d(all_lines, sections)
            result["report_lines"].extend(c_report)
            result["rotate_candidates"] = d_report
            result["report_lines"].extend(d_report)
            return result

    c_report, c_remove, c_archive = check_c(all_lines, sections, rel_path)
    d_report, d_remove, d_archive = check_d(all_lines, sections)
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

    existing_remove |= b_remove
    f_remove = set()
    f_archive = []
    for rng, prefix, chunk in f_removal_specs:
        if rng & existing_remove:
            continue  # already being rotated by Check B/C/D — avoid double-archiving
        f_remove |= rng
        f_archive.append((prefix, chunk))

    remove = existing_remove | f_remove
    archive_entries = (
        [(None, chunk) for chunk in c_archive]
        + [
            ("[resolved-watchpoint]", chunk)
            for rng, chunk in d_archive
            if not rng & c_remove
        ]
        + b_archive
        + f_archive
    )

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

    if archive_entries:
        archive_path = os.path.join(vault_root, ARCHIVE_DIR, f"{today_str[:7]}.md")
        header = f"## Rotated {today_str} from {rel_path}\n\n"
        pieces = [
            f"{prefix} {today_str}\n{chunk}" if prefix else chunk
            for prefix, chunk in archive_entries
        ]
        chunk_text = header + "\n".join(pieces) + "\n\n"
        # Shared cross-process mutation authority -- read-append-replace
        # under the same lock atomic_write uses, so two concurrent
        # appenders (or a concurrent MCP write) can never interleave bytes.
        try:
            append_archive_losslessly(archive_path, chunk_text)
        except Exception as exc:
            error = f"{rel_path}: archive write/preservation verification failed ({exc})"
            result["safety_exceptions"].append(error)
            result["operation_failed"] = True
            result["report_lines"].append(f"SAFETY-EXCEPTION: {error}")
            return result
        result["report_lines"].append(
            f"Archived {len(archive_entries)} bullet block(s) to {os.path.relpath(archive_path, vault_root)}"
        )

    try:
        write_file(abs_path, new_text)
        if read_file(abs_path) != new_text:
            raise OSError("post-write byte verification failed")
    except Exception as exc:
        error = f"{rel_path}: curated target write/verification failed ({exc})"
        result["safety_exceptions"].append(error)
        result["operation_failed"] = True
        result["report_lines"].append(f"SAFETY-EXCEPTION: {error}")
        return result
    result["changed"] = True

    if d_remove:
        result["report_lines"].append(
            f"Archived {len(d_report)} resolved watchpoint bullet(s) before removal"
        )

    return result


def process_all_targets(vault_root, apply_mode, today_str, now_ts):
    return [
        process_target(vault_root, rel, apply_mode, today_str, now_ts)
        for rel in TARGETS
    ]


def quiet_wait_seconds(vault_root, skipped_rel_paths, now_ts):
    """Calculate from fresh mtimes, never from the first-pass snapshot."""
    remaining = 0.0
    for rel_path in skipped_rel_paths:
        abs_path = os.path.join(vault_root, rel_path)
        if not os.path.isfile(abs_path):
            continue
        age = max(0.0, now_ts - os.path.getmtime(abs_path))
        remaining = max(remaining, max(0.0, MTIME_GUARD_SECONDS - age))
    return remaining + CRON_RETRY_MARGIN_SECONDS


def apply_with_bounded_retry(
    vault_root, today_str, invoked_by_cron, *, clock=time.time, sleep=time.sleep
):
    """Run apply, self-healing recent-mtime skips only for the cron path.

    Every retry re-reads and re-evaluates every target. Attempt diagnostics
    are returned for the dated report; callers must compute notification
    authority from the final state plus unresolved operation failures only.
    """
    start = clock()
    deadline = start + CRON_RETRY_LIMIT_SECONDS
    current = process_all_targets(vault_root, True, today_str, start)
    attempts = [current]

    def unresolved_operation_failures(results):
        return [
            error
            for result in results if result.get("operation_failed")
            for error in result["safety_exceptions"]
        ]

    if not invoked_by_cron:
        return attempts, current, unresolved_operation_failures(current)

    while True:
        skipped = [result["rel_path"] for result in current if result["mtime_skipped"]]
        if not skipped:
            return attempts, current, unresolved_operation_failures(current)

        now_ts = clock()
        wait_seconds = quiet_wait_seconds(vault_root, skipped, now_ts)
        if wait_seconds <= 0:
            wait_seconds = CRON_RETRY_MARGIN_SECONDS
        if now_ts + wait_seconds > deadline:
            targets = ", ".join(skipped)
            failures = unresolved_operation_failures(current)
            failures.append(
                f"quiet-window retry exhausted after {CRON_RETRY_LIMIT_SECONDS}s: {targets}"
            )
            return attempts, current, failures

        sleep(wait_seconds)
        current = process_all_targets(vault_root, True, today_str, clock())
        attempts.append(current)


def exception_message(today_str, exceptions, report_path):
    unique = list(dict.fromkeys(exceptions))
    lines = [f"hot-md-curate integrity exception — {today_str}"]
    lines.extend(f"- {error}" for error in unique[:4])
    if len(unique) > 4:
        lines.append(f"- plus {len(unique) - 4} more; see dated report")
    lines.append(f"Report: {report_path}")
    return "\n".join(lines)


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
        return 0

    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    invoked_by_cron = args.from_cron or os.environ.get("HOT_MD_CURATE_FROM_CRON") == "1"
    safety_exceptions = []
    attempts = []
    backup_dir_before = None
    backup_dir_after = None
    ts = now.strftime("%Y-%m-%dT%H-%M-%SZ")

    if apply_mode:
        try:
            backup_dir_before = backup_targets(args.vault_root, ts, "before")
        except Exception as exc:
            safety_exceptions.append(f"before-backup failed; apply was not started ({exc})")

    if apply_mode and not safety_exceptions:
        attempts, _last_apply_results, operation_failures = apply_with_bounded_retry(
            args.vault_root, today_str, invoked_by_cron
        )
        safety_exceptions.extend(operation_failures)
        try:
            backup_dir_after = backup_targets(args.vault_root, ts, "after")
        except Exception as exc:
            safety_exceptions.append(f"after-backup failed ({exc})")

    # Final notification authority and totals come from a fresh, non-mutating
    # read after all bounded apply/retry work. First-pass data is diagnostic.
    all_results = process_all_targets(args.vault_root, False, today_str, time.time())
    safety_exceptions.extend(
        error for result in all_results for error in result["safety_exceptions"]
    )

    report_lines = [
        f"# Hot.md Curation Report — {today_str}",
        "",
        f"Mode: {'apply' if apply_mode else 'report'}",
    ]
    if backup_dir_before:
        report_lines.append(f"Backup (before): {backup_dir_before}")
    if backup_dir_after:
        report_lines.append(f"Backup (after): {backup_dir_after}")
    report_lines.append("")

    if attempts:
        report_lines.append("## Apply attempts (diagnostic only)")
        for attempt_number, attempt in enumerate(attempts, 1):
            report_lines.append(f"### Attempt {attempt_number}")
            for result in attempt:
                report_lines.append(f"- {result['rel_path']}")
                for line in result["report_lines"]:
                    report_lines.append(f"  - {line}")
        report_lines.append("")

    report_lines.extend(["## Final state (notification authority)", ""])
    total_resolved = 0
    total_stale = 0
    total_rotate = 0
    total_content_stale = 0
    total_canonical_conflict = 0

    for res in all_results:
        report_lines.append(f"### {res['rel_path']}")
        if not res["exists"]:
            report_lines.extend(["- MISSING", ""])
            continue
        for line in res["report_lines"]:
            report_lines.append(f"- {line}")
        report_lines.append("")

        resolved = sum(1 for finding in res["resolved_findings"] if finding.startswith("RESOLVED"))
        stale = sum(1 for finding in res["resolved_findings"] if finding.startswith("STALE-BLOCKER"))
        total_resolved += resolved
        total_stale += stale
        total_rotate += len(res["rotate_candidates"])
        total_content_stale += 1 if res.get("content_stale") else 0
        total_canonical_conflict += len(res.get("canonical_conflicts") or [])

    totals_line = (
        f"Totals: RESOLVED={total_resolved} STALE-BLOCKER={total_stale} "
        f"ROTATE-CANDIDATE={total_rotate} CONTENT-STALE={total_content_stale} "
        f"CANONICAL-CONFLICT={total_canonical_conflict}"
    )
    report_lines.extend(["## Totals", f"- {totals_line}", ""])

    review_due, review_warnings = check_e(args.vault_root, today_str)
    report_lines.append("## Scheduled Reviews")
    report_lines.extend(f"- {line}" for line in review_due)
    report_lines.extend(f"- {line}" for line in review_warnings)
    if not review_due and not review_warnings:
        report_lines.append("- None due")
    report_lines.append("")

    if safety_exceptions:
        report_lines.append("## Unresolved safety / integrity exceptions")
        report_lines.extend(f"- {error}" for error in dict.fromkeys(safety_exceptions))
        report_lines.append("")

    report_text = "\n".join(report_lines) + "\n"
    report_path = os.path.join(
        args.vault_root,
        "BS 2nd Brain/Alcove/Infrastructure/hot-md-reports",
        f"{today_str}.md",
    )
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        write_file(report_path, report_text)
    except Exception as exc:
        safety_exceptions.append(f"dated report write failed ({exc})")

    print(report_text)
    if not any(error.startswith("dated report write failed") for error in safety_exceptions):
        print(f"Report written to {report_path}")

    if args.no_telegram:
        pass
    elif not invoked_by_cron:
        print("Telegram notify skipped: not invoked by cron (manual/build-invoked run)")
    elif not safety_exceptions:
        print("Telegram notify skipped: no unresolved safety/integrity exception")
    else:
        try:
            response = send_telegram(exception_message(today_str, safety_exceptions, report_path))
            print(f"Telegram response: {response}")
        except Exception as exc:
            print(f"Telegram send failed: {exc}", file=sys.stderr)

    return 1 if safety_exceptions else 0


if __name__ == "__main__":
    sys.exit(main())
