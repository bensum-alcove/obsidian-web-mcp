import importlib.util
import os
import stat
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "vault_safety_sweep.py"


def _load():
    spec = importlib.util.spec_from_file_location("vault_safety_sweep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_known_signature_repair_only_inserts_newline():
    sweep = _load()
    damaged = b"---\nupdated: '2026-08-15'---\n\n# Body\n---\n"

    repaired, evidence = sweep.known_signature_repair(damaged)

    assert repaired == b"---\nupdated: '2026-08-15'\n---\n\n# Body\n---\n"
    assert len(repaired) == len(damaged) + 1
    assert evidence["byte_offset"] == damaged.index(b"---", 4)


def test_non_signature_damage_is_not_repaired():
    sweep = _load()
    assert sweep.known_signature_repair(b"---\nupdated: [broken\n---\nbody\n") is None
    assert sweep.known_signature_repair(b"---\nnotes: |---\nbody\n") is None


def test_dry_run_is_byte_identical_and_reports_mode(tmp_path):
    sweep = _load()
    target = tmp_path / "damaged.md"
    original = b"---\r\nupdated: '2026-08-15'---\r\nBody\r\n"
    target.write_bytes(original)
    os.chmod(target, 0o600)

    report = sweep.run({"test": tmp_path})

    vault = report["vaults"][0]
    assert vault["frontmatter_errors"][0]["known_signature"] is True
    assert vault["permission_errors"][0]["mode"] == "0o600"
    assert target.read_bytes() == original
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_repairs_content_and_only_adds_other_read(tmp_path):
    sweep = _load()
    target = tmp_path / "damaged.md"
    target.write_bytes(b"---\nupdated: value---\nbody\n")
    os.chmod(target, 0o650)

    report = sweep.run(
        {"test": tmp_path}, repair_content=True, repair_permissions=True
    )

    assert target.read_bytes() == b"---\nupdated: value\n---\nbody\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o654
    assert all(item["verified"] for item in report["vaults"][0]["repairs"])


def test_symlink_is_reported_and_never_followed(tmp_path):
    sweep = _load()
    outside = tmp_path.parent / "outside.md"
    outside.write_text("---\nupdated: value---\nbody\n")
    link = tmp_path / "link.md"
    link.symlink_to(outside)

    report = sweep.run(
        {"test": tmp_path}, repair_content=True, repair_permissions=True
    )

    assert report["vaults"][0]["symlinks"] == ["link.md"]
    assert outside.read_text() == "---\nupdated: value---\nbody\n"


def test_content_repair_refuses_same_inode_content_change(tmp_path):
    sweep = _load()
    target = tmp_path / "note.md"
    original = b"---\nupdated: value---\nbody\n"
    repaired = b"---\nupdated: value\n---\nbody\n"
    target.write_bytes(original)
    opened = target.stat()
    mode = stat.S_IMODE(opened.st_mode)
    target.write_bytes(b"concurrent edit")

    with pytest.raises(RuntimeError, match="content changed"):
        sweep._atomic_replace_if_unchanged(
            target, opened, original, repaired, mode, {}
        )
    assert target.read_bytes() == b"concurrent edit"


def test_content_repair_refuses_concurrent_mode_change(tmp_path):
    sweep = _load()
    target = tmp_path / "note.md"
    original = b"---\nupdated: value---\nbody\n"
    repaired = b"---\nupdated: value\n---\nbody\n"
    target.write_bytes(original)
    os.chmod(target, 0o600)
    opened = target.stat()
    os.chmod(target, 0o604)

    with pytest.raises(RuntimeError, match="mode changed"):
        sweep._atomic_replace_if_unchanged(
            target, opened, original, repaired, 0o600, {}
        )
    assert stat.S_IMODE(target.stat().st_mode) == 0o604
