#!/usr/bin/env python3
"""vault_backup_selector.py — pick the newest known-good generation of a
vault's git-backed backup history (vault-chaos-recovery-suite build,
scenario #10 in mcp-infrastructure's chaos/failure-matrix.md: "corrupt
latest backup generation").

Each vault (BS/CB/Alcove Brain) IS its own git repository, backed up by
scripts/vault-backup.sh (in mcp-infrastructure) committing+pushing every 15
minutes. This module answers "if the newest commit's content is corrupt, how
far back do we have to go to find a generation that actually parses?" --
read-only, via `git show`/`git ls-tree` against the repo's object store, so
it never touches the working tree and never mutates history. Safe to point
at a real vault repo (it makes no writes at all), though every test here
uses a throwaway, git-init'd fixture repo, never a real vault.

Not wired into vault-backup.sh's live cron path by this build -- this spec's
own frontmatter says the deliverable is a drill harness proving the
capability, not a new always-on recovery service; wiring this into the
automated recovery path is a follow-up candidate, not silently done here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obsidian_vault_mcp.frontmatter_safe import FrontmatterError, parse_frontmatter  # noqa: E402


class GitCommandError(RuntimeError):
    pass


def _git(args: list[str], repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise GitCommandError(
            f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


@dataclass
class GenerationCheck:
    commit: str
    files_checked: int
    errors: list[dict]

    @property
    def clean(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return asdict(self)


def check_generation(repo_path: Path, commit: str) -> GenerationCheck:
    """Integrity-check every .md file's frontmatter at one commit, read-only
    (via `git show`, never a checkout)."""
    names = _git(["ls-tree", "-r", "--name-only", commit], repo_path).splitlines()
    md_files = [n for n in names if n.endswith(".md")]

    errors = []
    for rel in md_files:
        try:
            raw = _git(["show", f"{commit}:{rel}"], repo_path)
        except GitCommandError as exc:
            errors.append({"path": rel, "error": f"git show failed: {exc}"})
            continue
        if not raw.startswith("---\n") and not raw.startswith("---\r\n"):
            continue  # no frontmatter block to validate -- not itself an error
        try:
            parse_frontmatter(raw)
        except FrontmatterError as exc:
            errors.append({"path": rel, "error": str(exc)})

    return GenerationCheck(commit=commit, files_checked=len(md_files), errors=errors)


def select_known_good_commit(repo_path: Path, max_generations: int = 20) -> dict:
    """Walk `git log` backwards from HEAD, integrity-checking each generation.
    Returns the first (newest) clean commit, flagging every corrupt one it
    skipped along the way -- never silently returns a corrupt HEAD as the
    answer, and never claims a generation is clean without having checked it."""
    shas = _git(["log", "--format=%H", "-n", str(max_generations)], repo_path).strip().splitlines()

    checked = []
    flagged_corrupt = []
    selected = None
    for sha in shas:
        generation = check_generation(repo_path, sha)
        checked.append(generation.to_dict())
        if generation.clean:
            selected = sha
            break
        flagged_corrupt.append(sha)

    return {
        "repo_path": str(repo_path),
        "generations_walked": len(checked),
        "checked": checked,
        "flagged_corrupt": flagged_corrupt,
        "selected": selected,
        "head_was_corrupt": bool(shas) and shas[0] in flagged_corrupt,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--max-generations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = select_known_good_commit(args.repo_path, max_generations=args.max_generations)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0 if report["selected"] else 1


if __name__ == "__main__":
    sys.exit(main())
