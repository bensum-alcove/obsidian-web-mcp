#!/usr/bin/env python3
"""canonical_state_scan.py -- report-only duplicate-authority scanner
(build: vault-canonical-state-model).

Walks BS 2nd Brain/Alcove/Infrastructure/Canonical State/records/ and reports:
  - malformed records (frontmatter validation errors -- see
    obsidian_vault_mcp.canonical_state.validate_frontmatter)
  - duplicate authority: more than one non-superseded ("current") record
    claiming the same component_id

Never writes anything, never autofixes anything -- there is no --apply flag,
by design. This is a detector, same role as contradiction_lint.py plays for
SYSTEM-FACTS.md: it turns a data-integrity bug into a visible, mechanically
reproducible finding instead of a silent ambiguity the next reader has to
guess about. Exit code is 1 if any malformed record or duplicate authority is
found, 0 otherwise, so it can be wired into a cron/CI check without a human
reading the JSON every time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obsidian_vault_mcp import config  # noqa: E402
from obsidian_vault_mcp.canonical_state import (  # noqa: E402
    load_all_records,
    scan_duplicate_authority,
)

DEFAULT_RECORDS_DIR = (
    config.VAULT_PATH
    / "BS 2nd Brain"
    / "Alcove"
    / "Infrastructure"
    / "Canonical State"
    / "records"
)


def run(records_dir: Path | None = None) -> dict:
    records_dir = Path(records_dir) if records_dir is not None else DEFAULT_RECORDS_DIR
    _records, errors = load_all_records(records_dir)
    duplicates = scan_duplicate_authority(records_dir)
    return {
        "records_dir": str(records_dir),
        "malformed_records": errors,
        "duplicate_authority": {
            component_id: [str(r.path) for r in records]
            for component_id, records in duplicates.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run(args.records_dir)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if report["malformed_records"] or report["duplicate_authority"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
