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
SYSTEM-FACTS.md — those three files are never modified, by any code path, on
any run, findings or not. The only file this script writes is
contradiction-lint-report.md: the prior manual infrastructure.md analysis
there is carried forward verbatim (this script cannot redo that
reasoning-based pass without an LLM) under its own always-2026-06-30-dated
heading, and only the automated SYSTEM-FACTS section below it regenerates
each run, under its own separately-dated heading.

Matching is evidence-based, not co-occurrence-based (see contradiction-lint-
precision-fix, 2026-08-09): a fact only gets flagged if a changelog entry, in
a bounded window around a shared token, states what actually changed (an
explicit change signal, or a differing value for the same key) — a bare
shared token is not evidence. High-frequency infra nouns (`.wslconfig`,
`vault_search`, ...) need a second independent corroborating signal beyond
the noun itself, since they're mentioned in nearly every entry regardless of
topic. If a match can't state a concrete current truth, it isn't flagged.
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

# These tokens clear the stoplist (long enough, have a path/file separator) but
# still show up in nearly every infra changelog entry regardless of topic —
# ".wslconfig", "vault_search" etc. are mentioned constantly without the
# underlying fact they'd contradict actually changing. Co-occurrence with one
# of these alone is not evidence; require a second, independent corroborating
# signal (a distinct token from the same fact, or a matching value/quantity)
# before letting a match through. See Phase 1 of the precision-fix spec.
_HIGH_FREQ_NOUNS = {
    ".wslconfig", "supervisord.conf", "conf.d", "build-orchestrator",
    "vault_search", "vault_query", "check-supervisord.sh",
}
# Every `vault_*` MCP tool name (not just vault_search/vault_query above) is
# equally over-mentioned in this vault's own engineering changelog — any
# retrieval/tooling entry discusses several of them together as a matter of
# course, regardless of whether "MCP tools available" itself changed.
_HIGH_FREQ_PREFIXES = ("vault_",)

# Bounded window (chars) searched around a token occurrence for a change
# signal or differing value — deliberately small. Matching anywhere in a
# multi-thousand-word changelog entry is exactly the co-occurrence bug this
# fix exists to remove; the entry must say what changed *near* the token. Even
# a single "|"-delimited table row can span 150+ chars across unrelated
# columns (e.g. a "before" cell listing an untouched env var next to an
# "after" cell describing a *different* var being added) — 100 keeps the
# window inside roughly one clause/cell without crossing into the next.
_WINDOW_RADIUS = 100

# Phrases indicating the changelog entry states something actually changed,
# not just that the token was mentioned. "at least" per the spec — deliberately
# excludes weaker/ambiguous words like "instead of" or bare "never" ("X was
# never modified" reads as confirmation, not change, and must NOT count).
# "→" (also spec-listed) is deliberately excluded: this vault's changelog uses
# it constantly for unrelated things (HTTP status asides, boot-chain arrows,
# before/after table cells) that have nothing to do with the token being
# checked, so at any usable window size it reads as noise, not signal.
_CHANGE_SIGNAL_RE = re.compile(
    r"\b(moved|renamed|removed|replaced|superseded|deprecat\w*|added|now|no longer)\b",
    re.IGNORECASE,
)

# A value's own "shape" — used to check whether a changelog entry restates the
# *same* value (confirmation, no contradiction) or a *different* one (real
# evidence) for the same fact. Deliberately tied to a key word (memory/cap/
# ram) within a short span of the number — a bare "~20GB" floating in an
# unrelated sentence ("ruling out the obvious ~20GB hypothesis") is not a
# value for the fact's key, it just happens to share a unit.
_KEY_QUANTITY_RE = re.compile(
    r"(?:\b(?:memory|cap|ram)\b.{0,20}?(\d+(?:\.\d+)?)\s?(GB|GiB|MB|KB))"
    r"|(?:(\d+(?:\.\d+)?)\s?(GB|GiB|MB|KB).{0,20}?\b(?:memory|cap|ram)\b)",
    re.IGNORECASE,
)

# Characters that continue a path/identifier — used to tell a "maximal" token
# occurrence (its own free-standing identifier) from one merely embedded inside
# a longer one (e.g. the token `conf.d` inside a path ending in
# `conf.d/playwright-mcp.conf` describes a *different, more specific* thing
# than the fact's own value and must not count as evidence for it).
_IDENT_CONTINUATION = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-~/")


def _is_isolated_occurrence(entry_lower: str, start: int, end: int) -> bool:
    if start > 0 and entry_lower[start - 1] in _IDENT_CONTINUATION:
        return False
    if end < len(entry_lower) and entry_lower[end] in _IDENT_CONTINUATION:
        return False
    return True


def _quantities(text: str) -> set[tuple[str, str]]:
    out = set()
    for m in _KEY_QUANTITY_RE.finditer(text):
        val = m.group(1) or m.group(3)
        unit = m.group(2) or m.group(4)
        out.add((val, unit.lower().rstrip("s")))
    return out


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


def _self_referential(entry_text: str) -> bool:
    """Entries this lint itself generated (weekly SYSTEM-FACTS findings summaries)
    describe lint output, not an infrastructure change — they must not feed back
    in as evidence for the next run's matching, or the lint corroborates itself."""
    title = entry_text.strip().splitlines()[0] if entry_text.strip() else ""
    return "vault-fact-lint-recurring:" in title and "SYSTEM-FACTS contradiction/staleness findings" in title


def _find_contradiction(
    search_text: str,
    last_verified: str | None,
    changelog_entries: list[tuple[str | None, str]],
    now: datetime,
    is_correction: bool = False,
) -> dict | None:
    """First changelog entry, dated after `last_verified` (or within STALE_DAYS if
    there's no verification date at all), that gives *evidence* of a changed
    value for `search_text` — not merely a shared token. A bare token match is
    not evidence: the entry must state what changed, near the token, either via
    an explicit change signal (moved/renamed/removed/.../added/now/no longer) or
    a differing value/quantity for the same key. High-frequency infra nouns
    (`.wslconfig`, `vault_search`, ...) are mentioned in nearly every entry
    regardless of topic, so a plain fact needs a *second* independent
    corroborating signal (another distinct fact token in the same window, or a
    genuinely differing same-key value) beyond the noun itself — and an entry
    that explicitly restates the fact's own value unchanged vetoes the match
    outright, overriding any signal word elsewhere in the window.

    Correction Log rows get one deliberate exception to that veto: they record
    a fact Ben has already gotten wrong once, so any entry with a real change
    signal that touches the exact same config/file is worth flagging for a
    second look even if the specific value it restates is unchanged (e.g. a new
    section was added to the same file). Plain facts don't get this exception —
    that would be exactly the co-occurrence-as-evidence bug this rewrite exists
    to remove. See Phase 1/2 of the precision-fix spec.

    If no window can state a concrete current truth, no hit is returned: "if
    the tool cannot state what the truth now is, it must not flag."."""
    tokens = _extract_tokens(search_text)
    if not tokens:
        return None
    fact_quantities = _quantities(search_text)
    fact_units = {u for _, u in fact_quantities}
    for date_str, entry_text in changelog_entries:
        if not date_str:
            continue
        if last_verified:
            if date_str <= last_verified:
                continue
        elif _days_since(date_str, now) > STALE_DAYS:
            continue
        if _self_referential(entry_text):
            continue
        entry_lower = entry_text.lower()
        for tok in tokens:
            tok_lower = tok.lower()
            is_high_freq = any(noun in tok_lower for noun in _HIGH_FREQ_NOUNS) or tok_lower.startswith(
                _HIGH_FREQ_PREFIXES
            )
            search_from = 0
            while True:
                idx = entry_lower.find(tok_lower, search_from)
                if idx == -1:
                    break
                end = idx + len(tok_lower)
                search_from = end
                if not _is_isolated_occurrence(entry_lower, idx, end):
                    continue
                win_start, win_end = max(0, idx - _WINDOW_RADIUS), min(len(entry_text), end + _WINDOW_RADIUS)
                window = entry_text[win_start:win_end]

                # Only compare quantities that share a unit with something in
                # the fact — a "~5 min" restart cadence near the token is not
                # evidence about a fact whose own quantities are all GB/GiB.
                window_quantities = {(v, u) for v, u in _quantities(window) if u in fact_units}
                differing_quantities = window_quantities - fact_quantities
                same_quantity_only = bool(window_quantities) and not differing_quantities

                if same_quantity_only and not is_correction:
                    # Entry restates the same value near this token — that's
                    # confirmation, not contradiction. Corrections get the
                    # exception described above; plain facts don't.
                    continue

                has_signal = bool(_CHANGE_SIGNAL_RE.search(window))

                if is_high_freq and not is_correction:
                    # The noun alone (even with a differing quantity) isn't
                    # enough for a high-frequency noun on a plain fact — it
                    # needs an explicit change signal *and* a second
                    # independent corroborating signal (a differing value, or
                    # another distinct fact token, in the same window). A
                    # sibling from the same "family" (e.g. `vault_search` /
                    # `vault_semantic_search` — a fact enumerating many
                    # `vault_*` tools) doesn't count: those co-occur constantly
                    # in ordinary engineering prose about the vault codebase
                    # and aren't independent evidence of each other.
                    tok_prefix = re.split(r"[_.\-]", tok_lower, maxsplit=1)[0]
                    other_tokens_present = any(
                        t.lower() in window.lower()
                        for t in tokens
                        if t.lower() != tok_lower
                        and re.split(r"[_.\-]", t.lower(), maxsplit=1)[0] != tok_prefix
                    )
                    if not (has_signal and (differing_quantities or other_tokens_present)):
                        continue
                elif not (differing_quantities or has_signal):
                    continue

                title = entry_text.strip().splitlines()[0].lstrip("#").strip()
                title = re.sub(rf"^{re.escape(date_str)}\s*—\s*", "", title)
                snippet = " ".join(window.split())[:220]
                return {
                    "date": date_str,
                    "excerpt": title[:200],
                    "token": tok,
                    "current_truth": snippet,
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
                        "current_truth": f"{hit['current_truth']} (changelog {hit['date']})",
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
            hit = _find_contradiction(correct_fact, last_verified, changelog_entries, now, is_correction=True)
            if hit:
                correction_superseded.append(
                    {
                        "fact": (row.get(wrong_key, "") if wrong_key else "") + " → " + correct_fact,
                        "section": "Correction Log",
                        "changelog": hit["excerpt"],
                        "date": hit["date"],
                        "current_truth": f"Correction itself may be stale — {hit['current_truth']} (changelog {hit['date']})",
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
        SYSTEM_FACTS_MARKER,
        "",
        f"## Automated SYSTEM-FACTS analysis (generated {now.strftime('%Y-%m-%d')})",
        "",
        "Automated, weekly, by `scripts/contradiction_lint.py`. Independent of the manual "
        "infrastructure.md analysis above — that pass is one-time (2026-06-30) and not "
        "re-run by this script; only this section regenerates each run.",
        "",
        "### SYSTEM-FACTS contradictions",
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
    lines += ["", "### SYSTEM-FACTS stale (no confirmation >90d)", ""]
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
    lines += ["", "### SYSTEM-FACTS current", "", f"Count: {len(result['current'])}"]
    return "\n".join(lines) + "\n"


# Invisible cut-point for splitting "manual analysis" from "automated section" on
# rerun — an HTML comment so it doesn't clutter the rendered report the way the
# old inline italic note did, while still giving build_report() an exact,
# stable string to find. The old inline-italic marker is recognized too, purely
# so the one run that transitions a report from the old format to this one
# doesn't duplicate the carried-forward manual section.
SYSTEM_FACTS_MARKER = "<!-- SYSTEM_FACTS_AUTOMATED_SECTION -->"
_LEGACY_SYSTEM_FACTS_MARKER = "*SYSTEM-FACTS sections below regenerated"

# The manual 2026-06-30 infrastructure.md analysis is carried forward verbatim
# (this script cannot redo that reasoning-based pass) but it goes stale exactly
# because it's never re-run — e.g. it still lists Playwright MCP's headless
# setting, which stopped applying once playwright-mcp was decommissioned
# entirely on 2026-08-09. Flagging it costs nothing and keeps a fresh
# `generated:` frontmatter date from implying the whole report was re-verified.
_MANUAL_ANALYSIS_LABEL_MARKER = "one-time pass, not re-run by automation"
_MANUAL_ANALYSIS_LABEL = (
    "\n\n**Manual infrastructure.md analysis — one-time pass, not re-run by automation.** "
    "The frontmatter `generated:` date above tracks only the automated section at the bottom "
    "of this report; this section keeps its own 2026-06-30 date and goes stale independently."
)

_KNOWN_OBSOLETE_NOTE_MARKER = "## Known-obsolete manual-analysis rows (flagged, not corrected)"
_KNOWN_OBSOLETE_NOTE = f"""
{_KNOWN_OBSOLETE_NOTE_MARKER}

The manual analysis above is carried forward verbatim and never re-run, so it can go stale on
its own timeline. Known cases (flagged here, not edited into the row above — see Hard
Constraints):

- **Playwright MCP: "Browser: Headless Chromium..." row** — `playwright-mcp` (the program this
  row describes) was decommissioned entirely on 2026-08-09; see
  `infrastructure-changelog.md` entry `playwright-mcp-decommission-cleanup`. The row's headless/
  display-flag claim no longer applies to anything running.
"""


def build_report(existing_text: str, system_facts_section: str, now: datetime) -> str:
    """Carry forward everything up to (and not including) a prior SYSTEM_FACTS_MARKER
    verbatim, bump the `generated:` frontmatter date, and append the freshly
    regenerated SYSTEM-FACTS section — idempotent across reruns. `generated:`
    describes only the automated section; the manual analysis keeps its own
    2026-06-30 date inline so a fresh `generated:` value never implies the
    carried-forward content was re-verified."""
    today = now.strftime("%Y-%m-%d")
    if existing_text.startswith("---"):
        end = existing_text.find("\n---", 3)
        fm_block, body = existing_text[: end + 4], existing_text[end + 4 :]
    else:
        fm_block, body = "", existing_text

    fm_block = re.sub(r"^generated:\s*.*$", f"generated: '{today}'", fm_block, flags=re.MULTILINE)

    cut = body.find("\n" + SYSTEM_FACTS_MARKER)
    if cut == -1:
        cut = body.find("\n" + _LEGACY_SYSTEM_FACTS_MARKER)
    if cut != -1:
        body = body[:cut]

    if _MANUAL_ANALYSIS_LABEL_MARKER not in body:
        body = re.sub(r"(^Generated: *\d{4}-\d{2}-\d{2})$", r"\1" + _MANUAL_ANALYSIS_LABEL, body, count=1, flags=re.MULTILINE)

    if _KNOWN_OBSOLETE_NOTE_MARKER not in body:
        body = body.rstrip() + "\n" + _KNOWN_OBSOLETE_NOTE

    return fm_block + body.rstrip() + "\n\n" + system_facts_section


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

    # Report-only: this run never touches infrastructure.md, SYSTEM-FACTS.md,
    # or infrastructure-changelog.md — only contradiction-lint-report.md is
    # written. An earlier version of this script also prepended a summary
    # entry to infrastructure-changelog.md on findings; that contradicted its
    # own "those three files are never modified" claim and the precision-fix
    # spec's hard constraint, so it's gone — findings live in the report only.
    if not dry_run:
        REPORT_PATH.write_text(new_report, encoding="utf-8")

    summary = {
        "generated": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contradicted": n_contra,
        "stale": n_stale,
        "current": len(result["current"]),
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
