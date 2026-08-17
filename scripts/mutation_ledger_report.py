#!/usr/bin/env python3
"""Incident-response CLI over the vault mutation ledger.

Read-only. Filters and prints ledger events (see mutation_ledger.py for the
event schema) -- e.g. "what touched this path in the last hour" or "show me
every reject/conflict since this incident started". Complements `git log`,
which shows *what changed*; this shows *which tool call* did it and whether
it was accepted.

Examples:
    uv run python scripts/mutation_ledger_report.py --path-prefix "Clients/" --result reject
    uv run python scripts/mutation_ledger_report.py --since 2026-08-17T21:00:00+00:00 --json
    uv run python scripts/mutation_ledger_report.py --vault-path /home/ben_sum/vaults/bs-brain --tool vault_delete
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obsidian_vault_mcp import mutation_ledger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger-dir", type=Path, default=None, help="Ledger directory (default: VAULT_MUTATION_LEDGER_DIR or <vault-path>/.mutation-ledger)")
    parser.add_argument("--vault-path", type=Path, default=None, help="Vault root, used to derive the default ledger dir when --ledger-dir is not given")
    parser.add_argument("--path-prefix", help="Only events whose path starts with this prefix")
    parser.add_argument("--tool", help="Only events from this tool, e.g. vault_write")
    parser.add_argument("--result", choices=["success", "reject", "conflict"])
    parser.add_argument("--since", help="ISO timestamp lower bound (inclusive, string comparison)")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table")
    args = parser.parse_args()

    ledger_dir = args.ledger_dir
    if ledger_dir is None and args.vault_path is not None:
        ledger_dir = args.vault_path / ".mutation-ledger"

    events = mutation_ledger.query_events(
        ledger_dir=ledger_dir,
        path_prefix=args.path_prefix,
        tool=args.tool,
        result=args.result,
        since=args.since,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps(events, indent=2, ensure_ascii=False))
        return 0

    if not events:
        print("No matching mutation events.")
        return 0

    header = f"{'timestamp':<32} {'result':<8} {'operation':<8} {'tool':<28} {'code':<24} path"
    print(header)
    print("-" * len(header))
    for e in events:
        print(
            f"{e.get('timestamp', ''):<32} {e.get('result', ''):<8} {e.get('operation', ''):<8} "
            f"{e.get('tool', ''):<28} {str(e.get('code') or ''):<24} {e.get('path', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
