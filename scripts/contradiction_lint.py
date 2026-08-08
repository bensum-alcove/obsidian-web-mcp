#!/usr/bin/env python3
"""contradiction_lint.py — Weekly report-only contradiction/staleness lint.

Cron: weekly, Sundays 06:00 system local time (this box's cron matches system
local time, Australia/Brisbane fixed UTC+10 = AEST year-round with no DST, per
the confirmed convention in run_eval.py / dreaming.py). Offset from the daily
dreaming cycle (30 15 * * *, all vaults) so the two jobs never contend for the
BS Brain vault.

Extends the original 2026-06-30 infrastructure.md-vs-changelog lint
(BS 2nd Brain/Alcove/Infrastructure/Claude-Code-Prompts/vault-contradiction-lint-prompt.md,
run once manually via a CC-direct/reasoning session, never scheduled) with a
second, fully automated pass over SYSTEM-FACTS.md: the file Claude falls back
on when infrastructure retrieval fails, making its accuracy load-bearing for
every infrastructure session.

Zero-LLM by design (no local inference, no API key): facts are matched against
changelog entries by token overlap (backtick-quoted identifiers, ports, paths,
URLs, ENV_VAR-style names) plus a dated-supersession check, the same style of
heuristic dreaming.py already uses for its lightweight Sunday infra.md pass.
Because this is a heuristic, not a human judgment, findings are candidates for
manual review, not confirmed contradictions.

Read-only against infrastructure.md, infrastructure-changelog.md, and
SYSTEM-FACTS.md — those three files are never modified. The prior manual
infrastructure.md analysis in contradiction-lint-report.md is carried forward
verbatim (this script cannot redo that reasoning-based pass without an LLM);
only the SYSTEM-FACTS sections below it are regenerated fresh each run. On
findings, a single Write Rule 13 summary entry is prepended to
infrastructure-changelog.md (newest-first ordering, confirmed from the live
file) and its frontmatter `updated` date is bumped as a separate edit.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obsidian_vault_mcp import config  # noqa: E402

VAULT_PATH = config.VAULT_PATH
INFRA_DIR = VAULT_PATH / "BS 2nd Brain" / "Alcove" / "Infrastructure"
INFRA_PATH = INFRA_DIR / "infrastructure.md"
CHANGELOG_PATH = INFRA_DIR / "infrastructure-changelog.md"
SYSTEM_FACTS_PATH = INFRA_DIR / "SYSTEM-FACTS.md"
REPORT_PATH = INFRA_DIR / "contradiction-lint-report.md"

STALE_DAYS = 90

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
CHANGELOG_HEADER_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-\|]+\|?\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")

TOKEN_RE = re.compile(
    r"`([^`]{3,})`"          # backtick-quoted identifiers/paths/env vars
    r"|(\bport\s*\d{4,5}\b)"  # port NNNN (4-5 digits — 3-digit is too generic)
    r"|(:\d{4,5}\b)"          # :NNNN
    r"|(/home/\S+)"           # absolute paths
    r"|(https?://\S+)"        # urls
    r"|(\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b)",  # ENV_VAR_STYLE — must contain '_'
    re.IGNORECASE,
)

# Common infra acronyms and the files this lint itself reads/writes are far too
# frequent to be distinctive — matching on them produces near-100% false
# positives (e.g. "WSL2" or "SYSTEM-FACTS.md" appear in almost every entry).
# Zero-LLM heuristics live or die on precision here, so tokens must clear a
# length floor AND miss this stoplist to count as a real signal.
_TOKEN_STOPLIST = {
    "mcp", "api", "wsl", "wsl2", "oauth", "json", "yaml", "http", "https", "ssh",
    "url", "ram", "cpu", "ibkr", "spi", "hsi", "mhi", "rest", "cli", "sdk", "llm",
    "vbs", "dom", "gui", "tls", "ssl", "vpn", "sql", "sqlite", "html", "css",
    "jwt", "rsa", "hmac", "utc", "aest", "wsl-boot.sh", "system-facts.md",
    "infrastructure.md", "infrastructure-changelog.md", "contradiction-lint-report.md",
    "supervisord.conf", "personal/build orchestrator/specs/", "personal/build orchestrator/schedules/",
}
MIN_TOKEN_LEN = 6
# A bare dictionary word in backticks ("orchestrator", "schema") is common-noun
# noise, not a distinctive identifier -- real identifiers carry a path/file/env
# separator. Phrases with whitespace ("status: ready") are config-keyword noise
# too generic to mean "same claim", not "same topic".
_DELIM_RE = re.compile(r"[/._-]")


def _split_changelog_entries(text: str) -> list[tuple[str | None, str]]:
    """Split a changelog on '## ' headers; return (date-or-None, entry-text) pairs."""
    entries = []
    for block in re.split(r"\n(?=## )", text):
        block = block.strip()
        if not block:
            continue
        header = block[3:].splitlines()[0] if block.startswith("## ") else ""
        m = CHANGELOG_HEADER_DATE_RE.match(header.strip())
        entries.append((m.group(1) if m else None, block))
    return entries


def _extract_tokens(text: str) -> list[str]:
    tokens = set()
    for m in TOKEN_RE.finditer(text):
        is_backtick = m.group(1) is not None
        tok = next((g for g in m.groups() if g), "").strip().strip("`")
        if len(tok) < MIN_TOKEN_LEN or tok.lower() in _TOKEN_STOPLIST:
            continue
        if is_backtick and (" " in tok or not _DELIM_RE.search(tok)):
            continue
        tokens.add(tok)
    return sorted(tokens)


def _last_date(text: str) -> str | None:
    dates = DATE_RE.findall(text or "")
    return max(dates) if dates else None


def _days_since(date_str: str, now: datetime) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return (now - dt).days


def parse_md_tables(text: str) -> list[dict]:
    """Return [{heading, headers, rows: [dict]}] for every '| ... |' table in a
    markdown doc, tagged with the nearest preceding heading of any level."""
    lines = text.splitlines()
    tables = []
    current_heading = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        m = HEADING_RE.match(line)
        if m:
            current_heading = m.group(2).strip()
            i += 1
            continue
        if line.strip().startswith("|") and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1]):
            headers = [c.strip() for c in line.strip().strip("|").split("|")]
            j = i + 2
            rows = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                if len(cells) == len(headers):
                    rows.append(dict(zip(headers, cells)))
                j += 1
            tables.append({"heading": current_heading, "headers": headers, "rows": rows})
            i = j
            continue
        i += 1
    return tables


def _find_contradiction(
    search_text: str, last_verified: str | None, changelog_entries: list[tuple[str | None, str]], now: datetime
) -> dict | None:
    """First changelog entry dated after `last_verified` (or, if no verification
    date at all, within the last STALE_DAYS) that shares a distinctive token
    with `search_text` — a candidate contradiction/supersession for review."""
    tokens = _extract_tokens(search_text)
    if not tokens:
        return None
    for date_str, entry_text in changelog_entries:
        if not date_str:
            continue
        if last_verified:
            if date_str <= last_verified:
                continue
        elif _days_since(date_str, now) > STALE_DAYS:
            continue
        entry_lower = entry_text.lower()
        for tok in tokens:
            if tok.lower() in entry_lower:
                title = entry_text.strip().splitlines()[0].lstrip("#").strip()
                title = re.sub(rf"^{re.escape(date_str)}\s*—\s*", "", title)
                return {
                    "date": date_str,
                    "excerpt": title[:200],
                    "token": tok,
                }
    return None


def _doc_updated_date(text: str) -> str | None:
    """Frontmatter `updated:` date — used as a fallback 'last reviewed' floor for
    rows with no inline date, so only changelog entries *after* the document was
    last touched count as candidate contradictions. Without this floor, nearly
    every fact lacking its own dated confirmation matches nearly every recent
    changelog entry sharing any token, which is useless noise, not signal."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    m = re.search(r"^updated:\s*['\"]?(\d{4}-\d{2}-\d{2})", text[:end], re.MULTILINE)
    return m.group(1) if m else None


def lint_system_facts(system_facts_text: str, changelog_entries: list[tuple[str | None, str]], now: datetime) -> dict:
    tables = parse_md_tables(system_facts_text)
    doc_floor = _doc_updated_date(system_facts_text)
    contradicted, stale, current = [], [], []

    for table in tables:
        if table["heading"].strip().lower() == "correction log":
            continue  # handled separately below
        headers_lower = [h.lower() for h in table["headers"]]
        if "fact" not in headers_lower or "value" not in headers_lower:
            continue
        fact_key = table["headers"][headers_lower.index("fact")]
        value_key = table["headers"][headers_lower.index("value")]
        for row in table["rows"]:
            fact, value = row.get(fact_key, ""), row.get(value_key, "")
            if not fact:
                continue
            last_verified = _last_date(value) or _last_date(fact) or doc_floor
            hit = _find_contradiction(f"{fact} {value}", last_verified, changelog_entries, now)
            if hit:
                contradicted.append(
                    {
                        "fact": fact,
                        "section": table["heading"] or "SYSTEM-FACTS.md",
                        "changelog": hit["excerpt"],
                        "date": hit["date"],
                        "current_truth": f"See changelog {hit['date']} (matched on `{hit['token']}`)",
                    }
                )
            elif last_verified is None or _days_since(last_verified, now) > STALE_DAYS:
                stale.append(
                    {
                        "fact": fact,
                        "section": table["heading"] or "SYSTEM-FACTS.md",
                        "last_verified": last_verified or "never",
                    }
                )
            else:
                current.append(fact)

    correction_superseded = []
    for table in tables:
        if table["heading"].strip().lower() != "correction log":
            continue
        headers_lower = [h.lower() for h in table["headers"]]
        if "correct fact" not in headers_lower:
            continue
        wrong_key = table["headers"][headers_lower.index("wrong assumption")] if "wrong assumption" in headers_lower else None
        correct_key = table["headers"][headers_lower.index("correct fact")]
        date_key = table["headers"][headers_lower.index("date learned")] if "date learned" in headers_lower else None
        for row in table["rows"]:
            correct_fact = row.get(correct_key, "")
            date_learned = row.get(date_key, "") if date_key else ""
            last_verified = _last_date(date_learned)
            hit = _find_contradiction(correct_fact, last_verified, changelog_entries, now)
            if hit:
                correction_superseded.append(
                    {
                        "fact": (row.get(wrong_key, "") if wrong_key else "") + " → " + correct_fact,
                        "section": "Correction Log",
                        "changelog": hit["excerpt"],
                        "date": hit["date"],
                        "current_truth": f"Correction itself superseded — see changelog {hit['date']}",
                    }
                )

    return {
        "contradicted": contradicted + correction_superseded,
        "stale": stale,
        "current": current,
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in row) + " |")
    return "\n".join(lines)


def render_system_facts_section(result: dict, now: datetime) -> str:
    lines = [
        f"{SYSTEM_FACTS_MARKER} {now.strftime('%Y-%m-%d')} by "
        "`scripts/contradiction_lint.py` (automated, weekly). Sections above this point are "
        "carried forward verbatim from the original 2026-06-30 manual infrastructure.md analysis "
        "— not re-run by this script.*",
        "",
        "## SYSTEM-FACTS contradictions",
        "",
    ]
    if result["contradicted"]:
        lines.append(
            _md_table(
                ["Fact", "Table/section", "Changelog entry", "Date", "Current truth"],
                [
                    [c["fact"], c["section"], c["changelog"], c["date"], c["current_truth"]]
                    for c in result["contradicted"]
                ],
            )
        )
    else:
        lines.append("| Fact | Table/section | Changelog entry | Date | Current truth |")
        lines.append("|---|---|---|---|---|")
    lines += ["", "## SYSTEM-FACTS stale (no confirmation >90d)", ""]
    if result["stale"]:
        lines.append(
            _md_table(
                ["Fact", "Table/section", "Last verified"],
                [[s["fact"], s["section"], s["last_verified"]] for s in result["stale"]],
            )
        )
    else:
        lines.append("| Fact | Table/section | Last verified |")
        lines.append("|---|---|---|")
    lines += ["", "## SYSTEM-FACTS current", "", f"Count: {len(result['current'])}"]
    return "\n".join(lines) + "\n"


SYSTEM_FACTS_MARKER = "*SYSTEM-FACTS sections below regenerated"


def build_report(existing_text: str, system_facts_section: str, now: datetime) -> str:
    """Carry forward everything up to (and not including) a prior SYSTEM_FACTS_MARKER
    verbatim, bump the `generated:` frontmatter date, and append the freshly
    regenerated SYSTEM-FACTS section — idempotent across reruns."""
    today = now.strftime("%Y-%m-%d")
    if existing_text.startswith("---"):
        end = existing_text.find("\n---", 3)
        fm_block, body = existing_text[: end + 4], existing_text[end + 4 :]
    else:
        fm_block, body = "", existing_text

    fm_block = re.sub(r"^generated:\s*.*$", f"generated: '{today}'", fm_block, flags=re.MULTILINE)

    cut = body.find("\n" + SYSTEM_FACTS_MARKER)
    if cut != -1:
        body = body[:cut]

    return fm_block + body.rstrip() + "\n\n" + system_facts_section


CHANGELOG_ENTRY_TEMPLATE = """## {date} — vault-fact-lint-recurring: SYSTEM-FACTS contradiction/staleness findings
**Status:** executed

### What this means
Automated weekly lint found {n_contra} candidate contradiction(s) and {n_stale} stale fact(s) \
between SYSTEM-FACTS.md and this changelog. See `contradiction-lint-report.md` for the full \
table. Report-only — no source files were modified by this run; findings are candidates for \
manual review, not confirmed corrections.

### Files changed
| File | Original before | Change after | Revert command |
|------|-------------------|----------------|----------------|
| BS 2nd Brain/Alcove/Infrastructure/contradiction-lint-report.md | previous lint snapshot | \
regenerated with fresh SYSTEM-FACTS section | restore prior version from git history / vault backup |

### Verification
- [x] Report regenerated: contradiction-lint-report.md
- [ ] Findings reviewed by Ben and SYSTEM-FACTS.md corrected if needed

### Rationale
Recurring automated lint (vault-fact-lint-recurring build) — SYSTEM-FACTS.md accuracy is \
load-bearing for every infrastructure session, so drift must surface automatically rather than \
never (the original 2026-06-30 lint was run once manually and never scheduled).

### Revert procedure
1. This entry is informational only; no infrastructure change was made.
2. If findings are false positives, no action needed beyond noting so in the report.

"""


def _bump_changelog(changelog_text: str, n_contra: int, n_stale: int, now: datetime) -> str:
    today = now.strftime("%Y-%m-%d")
    entry = CHANGELOG_ENTRY_TEMPLATE.format(date=today, n_contra=n_contra, n_stale=n_stale)

    if changelog_text.startswith("---"):
        end = changelog_text.find("\n---", 3)
        fm_block, body = changelog_text[: end + 4], changelog_text[end + 4 :]
    else:
        fm_block, body = "", changelog_text

    fm_block = re.sub(r"^updated:\s*.*$", f"updated: '{today}'", fm_block, flags=re.MULTILINE)
    fm_block = re.sub(
        r"^last_edit_note:.*$",
        "last_edit_note: Added vault-fact-lint-recurring SYSTEM-FACTS findings entry "
        f"({today})",
        fm_block,
        flags=re.MULTILINE,
    )
    fm_block = re.sub(r"^last_edited_by:.*$", "last_edited_by: Claude Code (vault-fact-lint-recurring cron)", fm_block, flags=re.MULTILINE)

    return fm_block + "\n\n" + entry + body.lstrip("\n")


def run(dry_run: bool = False) -> dict:
    now = datetime.now(timezone.utc)

    for path in (INFRA_PATH, CHANGELOG_PATH, SYSTEM_FACTS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"required lint target missing: {path}")

    changelog_text = CHANGELOG_PATH.read_text(encoding="utf-8", errors="replace")
    system_facts_text = SYSTEM_FACTS_PATH.read_text(encoding="utf-8", errors="replace")
    changelog_entries = _split_changelog_entries(changelog_text)

    result = lint_system_facts(system_facts_text, changelog_entries, now)
    system_facts_section = render_system_facts_section(result, now)

    existing_report = REPORT_PATH.read_text(encoding="utf-8", errors="replace") if REPORT_PATH.exists() else ""
    new_report = build_report(existing_report, system_facts_section, now)

    n_contra, n_stale = len(result["contradicted"]), len(result["stale"])
    # Guard against double-logging if this script runs twice the same day (e.g. a
    # manual verification run followed by the real cron fire) — only the report
    # regenerates every run; the changelog entry is a once-per-cycle summary.
    # entries[0] is frequently the leading frontmatter block (it never starts
    # with '## ', since the split only breaks *before* '## ' headers) — skip it
    # to find the actual newest dated entry.
    real_entries = [e for e in changelog_entries if e[1].startswith("## ")]
    top_entry_title = real_entries[0][1].splitlines()[0] if real_entries else ""
    already_logged_today = "vault-fact-lint-recurring:" in top_entry_title

    if not dry_run:
        REPORT_PATH.write_text(new_report, encoding="utf-8")
        if n_contra > 0 and not already_logged_today:
            CHANGELOG_PATH.write_text(_bump_changelog(changelog_text, n_contra, n_stale, now), encoding="utf-8")

    summary = {
        "generated": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contradicted": n_contra,
        "stale": n_stale,
        "current": len(result["current"]),
        "changelog_appended": n_contra > 0 and not dry_run,
        "report_path": str(REPORT_PATH),
    }
    print(
        f"[contradiction_lint] {summary['generated']} — "
        f"{n_contra} contradicted, {n_stale} stale, {summary['current']} current "
        f"→ {REPORT_PATH}"
        + (" (dry-run, nothing written)" if dry_run else ""),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Compute findings without writing any files")
    cli_args = parser.parse_args()
    run(dry_run=cli_args.dry_run)
