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


def _list_md_files(repo_path: Path, commit: str) -> list[str]:
    names = _git(["ls-tree", "-r", "--name-only", commit], repo_path).splitlines()
    return [n for n in names if n.endswith(".md")]


def check_generation(repo_path: Path, commit: str, md_files: list[str] | None = None) -> GenerationCheck:
    """Integrity-check every .md file's frontmatter at one commit, read-only
    (via `git show`, never a checkout)."""
    if md_files is None:
        md_files = _list_md_files(repo_path, commit)

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


def select_known_good_commit(
    repo_path: Path, max_generations: int = 20, min_completeness_ratio: float = 0.9
) -> dict:
    """Walk `git log` backwards from HEAD, integrity-checking each generation.
    Returns the first (newest) clean commit, flagging every corrupt one it
    skipped along the way -- never silently returns a corrupt HEAD as the
    answer, and never claims a generation is clean without having checked it.

    Frontmatter parseability alone can't catch a generation that silently
    lost most of the vault (e.g. a partial/failed sync) but still parses fine
    -- so this first surveys .md file counts across the whole walked window
    (cheap, `git ls-tree` only) to establish the vault's peak size in that
    window, then treats any generation whose file count falls below
    `min_completeness_ratio` of that peak as incomplete, on top of the
    per-file frontmatter check.
    """
    shas = _git(["log", "--format=%H", "-n", str(max_generations)], repo_path).strip().splitlines()

    file_lists = {sha: _list_md_files(repo_path, sha) for sha in shas}
    max_files_seen = max((len(v) for v in file_lists.values()), default=0)

    checked = []
    flagged_corrupt = []
    selected = None
    for sha in shas:
        generation = check_generation(repo_path, sha, md_files=file_lists[sha])
        expected_min = max(1, round(max_files_seen * min_completeness_ratio)) if max_files_seen else 0
        if generation.files_checked < expected_min:
            generation.errors.append({
                "path": "<generation-manifest>",
                "error": (
                    f"incomplete generation: {generation.files_checked} .md files, "
                    f"expected >= {expected_min} (>= {min_completeness_ratio:.0%} of "
                    f"{max_files_seen} seen across the walked window)"
                ),
            })
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
        "max_files_seen_in_window": max_files_seen,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--max-generations", type=int, default=20)
    parser.add_argument("--min-completeness-ratio", type=float, default=0.9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = select_known_good_commit(
        args.repo_path,
        max_generations=args.max_generations,
        min_completeness_ratio=args.min_completeness_ratio,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0 if report["selected"] else 1


if __name__ == "__main__":
    sys.exit(main())
