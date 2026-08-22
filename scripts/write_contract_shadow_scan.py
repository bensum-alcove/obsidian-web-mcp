#!/usr/bin/env python3
"""Shadow-scan real vaults against the write-contract gate's validators.

Read-only. Simulates "what would happen if this exact file were rewritten
as-is through vault_write / moved / deleted" for every file already on disk,
and reports rule/path hit counts. Never modifies a file. This is how a rule
gets proven safe enough to flip from shadow-only to enforced=True: zero (or
fully explained) hits across all three real vaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obsidian_vault_mcp import write_contract as wc  # noqa: E402

DEFAULT_VAULTS = {
    "bs-brain": Path("/home/ben_sum/vaults/bs-brain"),
    "alcove-brain": Path("/home/ben_sum/vaults/alcove-brain"),
    "cb-brain": Path("/home/ben_sum/vaults/cb-brain"),
}
EXCLUDED_DIRS = {".git", ".trash", ".obsidian", ".semantic-index"}


def scan_vault(name: str, root: Path) -> dict:
    report = {
        "vault": name,
        "root": str(root),
        "files_scanned": 0,
        "rule_hit_counts": {rule: 0 for rule in wc.registered_content_rules()},
        "path_rule_hit_counts": {rule: 0 for rule in wc.registered_path_rules()},
        "hits": [],  # capped detail list per rule below
        "operational_errors": [],
    }
    per_rule_examples: dict[str, list[str]] = {r: [] for r in wc.registered_content_rules()}
    per_path_rule_examples: dict[str, list[str]] = {r: [] for r in wc.registered_path_rules()}
    MAX_EXAMPLES = 25

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if EXCLUDED_DIRS & set(path.relative_to(root).parts):
            continue
        relative = str(path.relative_to(root))

        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            report["operational_errors"].append({"path": relative, "error": str(exc)})
            continue

        report["files_scanned"] += 1

        # Content rules: simulate the worst case, a whole-file vault_write of
        # this exact content back onto itself (old == new). This surfaces
        # every mechanically-detectable defect already present in the file
        # without ever proposing a real change to it.
        ctx = wc.WriteContext(path=relative, old_content=content, new_content=content, tool="vault_write")
        for rule_id, fn in wc._CONTENT_REGISTRY.items():
            try:
                issues = fn(ctx)
            except Exception as exc:
                report["operational_errors"].append({"path": relative, "rule": rule_id, "error": str(exc)})
                continue
            if issues:
                report["rule_hit_counts"][rule_id] += 1
                if len(per_rule_examples[rule_id]) < MAX_EXAMPLES:
                    per_rule_examples[rule_id].append(relative)

        # Path rules: simulate a delete of every file (the more restrictive
        # of the two path operations) to see which existing paths a
        # protected-path rule would object to.
        path_ctx = wc.PathMutationContext(path=relative, operation="delete")
        for rule_id, fn in wc._PATH_REGISTRY.items():
            try:
                issues = fn(path_ctx)
            except Exception as exc:
                report["operational_errors"].append({"path": relative, "rule": rule_id, "error": str(exc)})
                continue
            if issues:
                report["path_rule_hit_counts"][rule_id] += 1
                if len(per_path_rule_examples[rule_id]) < MAX_EXAMPLES:
                    per_path_rule_examples[rule_id].append(relative)

    report["examples"] = {k: v for k, v in per_rule_examples.items() if v}
    report["path_examples"] = {k: v for k, v in per_path_rule_examples.items() if v}
    return report


def run(vaults: dict[str, Path] | None = None) -> dict:
    vaults = vaults or DEFAULT_VAULTS
    reports = [scan_vault(name, root) for name, root in vaults.items()]
    return {"mode": "dry-run", "auto_repair": False, "vaults": reports}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not any(v["operational_errors"] for v in report["vaults"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
