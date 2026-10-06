"""Static checks for scripts/verify-deploy.sh -- the BO deployment check.

Doesn't require live supervisor/network state (that's exercised by running
the script itself); just guards the read-only contract and syntax.
"""
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify-deploy.sh"


def test_exists_and_executable():
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & 0o111, "verify-deploy.sh must be executable"


def test_bash_syntax_is_valid():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


def test_uses_strict_mode_and_names_every_vault_program():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    for prog in ("bs-brain-vault", "cb-brain-vault", "alcove-brain-vault"):
        assert prog in text


def test_never_restarts_or_writes_outside_tmp():
    text = SCRIPT.read_text()
    forbidden = ("supervisorctl restart", "supervisorctl stop", "supervisorctl start", "rm -rf")
    for token in forbidden:
        assert token not in text, f"verify-deploy.sh must stay read-only ({token!r} found)"
