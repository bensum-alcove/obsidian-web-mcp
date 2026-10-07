"""Write Rule 21 helpers shared by scripts/dreaming.py and scripts/log-rollover.py.

Pure functions only -- nothing here touches a vault. See BS 2nd Brain/_SCHEMA.md
Write Rule 21: every hot.md is compiled truth under a char budget, and
append-only logs roll over.
"""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter

from . import config

# A log is flagged "rollover due" at this size. Write Rule 21 says 500KB; the
# brain-hygiene-v1 spec fixes the automatic flag at 900KB (decimal KB, so it
# fires slightly early rather than late).
LOG_ROLLOVER_BYTES = 900_000

# Append-only logs named by Write Rule 21, relative to the vault root. A path
# that does not exist in the vault being scanned is skipped, except the entries
# in REQUIRED_LOGS, which the dreaming report lists as "missing".
APPEND_ONLY_LOGS = (
    "BS 2nd Brain/Alcove/Infrastructure/infrastructure-changelog.md",
    "BS 2nd Brain/_log.md",
    "Personal/Trading System Changelog.md",
    "Personal/ETS-Learnings.md",
    "Personal/Build Orchestrator/reviews/theme-registry.md",
    "Compliance/Audit Learnings.md",
    "Alcove Brain/Compliance/Audit Learnings.md",
    "_log.md",
    "CB Brain/_log.md",
)
REQUIRED_LOGS = APPEND_ONLY_LOGS[:2]

DIARY_HEADING_RE = re.compile(r"^##\s+\d{4}-\d{2}-\d{2}\b", re.MULTILINE)


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (metadata, body); metadata is {} when absent or unparseable."""
    try:
        post = frontmatter.loads(text)
        return dict(post.metadata or {}), post.content
    except Exception:
        return {}, text


def hot_md_status(text: str) -> dict:
    """Size vs budget and diary-heading flags for one hot.md's text.

    chars counts the body only (frontmatter excluded). The budget is the
    frontmatter ``char_budget`` when it is a positive int, else the default.
    """
    meta, body = split_frontmatter(text)
    budget = config.HOT_MD_BUDGET_CHARS
    raw = meta.get("char_budget")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        budget = raw
    diary = DIARY_HEADING_RE.findall(body)
    return {
        "chars": len(body),
        "budget": budget,
        "over_budget": len(body) > budget,
        "diary_headings": len(diary),
    }


def log_status(size_bytes: int) -> dict:
    return {"bytes": size_bytes, "rollover_due": size_bytes >= LOG_ROLLOVER_BYTES}


def existing_logs(vault_path: Path) -> list[str]:
    return [rel for rel in APPEND_ONLY_LOGS if (vault_path / rel).is_file()]
