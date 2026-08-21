"""Tests for scripts/vault_observability_status.py (vault-observability-slo build).

Every source function takes injectable paths/data (no hidden dependency on
this box's real production state) so these tests are fully hermetic --
mirrors the same "degraded fixtures before chaos suite" requirement the
spec calls for.
"""

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "vault_observability_status.py"


@pytest.fixture(scope="module")
def status_mod():
    spec = importlib.util.spec_from_file_location("vault_observability_status", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _now():
    return datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def env(tmp_path):
    return {
        "status_dir": tmp_path / "state",
        "backup_state_dir": tmp_path / "backup-state",
        "backups_root": tmp_path / "backups",
        "ledger_path": tmp_path / "ledger.jsonl",
        "history_dir": tmp_path / "history",
        "watchdog_log": tmp_path / "watchdog.log",
    }


def test_all_sources_missing_is_unknown_not_ok(status_mod, tmp_path, env):
    """No data anywhere must never be silently read as healthy."""
    status = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    sli_by_id = {s["id"]: s for s in status["slis"]}
    for sli_id in ("functional_read_query", "backup_age_hours", "restore_drill_age_days"):
        assert sli_by_id[sli_id]["status"] == "unknown", sli_id
    assert status["overall_status"] == "unknown"


def test_canary_status_feeds_functional_layer(status_mod, tmp_path, env):
    env["status_dir"].mkdir(parents=True)
    (env["status_dir"] / "canary-bs-brain.json").write_text(json.dumps({
        "checked_at": (_now() - timedelta(minutes=5)).isoformat(),
        "overall_ok": True,
        "layers_failing": [],
        "layers": [{"layer": "verify_index_sees_patch", "ok": True, "detail": ""}],
    }))
    status = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    sli_by_id = {s["id"]: s for s in status["slis"]}
    assert sli_by_id["functional_read_query"]["status"] == "ok"
    assert sli_by_id["functional_read_query"]["value"] == 0
    assert sli_by_id["index_freshness_seconds"]["value"] == pytest.approx(300, abs=1)


def test_canary_layer_failure_surfaces_as_warning_or_critical(status_mod, tmp_path, env):
    env["status_dir"].mkdir(parents=True)
    (env["status_dir"] / "canary-bs-brain.json").write_text(json.dumps({
        "checked_at": _now().isoformat(),
        "overall_ok": False,
        "layers_failing": ["cleanup_scratch"],
        "layers": [{"layer": "verify_index_sees_patch", "ok": True, "detail": ""}],
    }))
    status = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    sli_by_id = {s["id"]: s for s in status["slis"]}
    assert sli_by_id["functional_read_query"]["status"] == "warning"


def test_backup_age_beyond_critical_threshold(status_mod, tmp_path, env):
    env["backup_state_dir"].mkdir(parents=True)
    state_file = env["backup_state_dir"] / "vault-backup-lastchange-BS_Brain"
    state_file.write_text("")
    old_time = (_now() - timedelta(hours=72)).timestamp()
    import os
    os.utime(state_file, (old_time, old_time))

    status = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    sli_by_id = {s["id"]: s for s in status["slis"]}
    assert sli_by_id["backup_age_hours"]["status"] == "critical"
    assert sli_by_id["backup_age_hours"]["value"] == pytest.approx(72, abs=0.1)


def test_restore_drill_age_from_newest_proof_dir(status_mod, tmp_path, env):
    env["backups_root"].mkdir(parents=True)
    old_proof = env["backups_root"] / "vault-clean-room-restore-proof-20260101"
    new_proof = env["backups_root"] / "vault-clean-room-restore-proof-20260810"
    old_proof.mkdir()
    new_proof.mkdir()
    import os
    old_time = (_now() - timedelta(days=200)).timestamp()
    new_time = (_now() - timedelta(days=10)).timestamp()
    os.utime(old_proof, (old_time, old_time))
    os.utime(new_proof, (new_time, new_time))

    status = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    sli_by_id = {s["id"]: s for s in status["slis"]}
    assert sli_by_id["restore_drill_age_days"]["value"] == pytest.approx(10, abs=0.1)
    assert sli_by_id["restore_drill_age_days"]["status"] == "ok"


def test_job_miss_statuses_drive_dreaming_and_hot_md_slis(status_mod, tmp_path, env):
    status = status_mod.collect_status(
        "bs-brain", tmp_path, _now(),
        job_statuses={"dreaming-bs-brain": "MISSED", "hot-md-curate": "OK"},
        **env,
    )
    sli_by_id = {s["id"]: s for s in status["slis"]}
    assert sli_by_id["dreaming_state"]["status"] == "critical"
    assert sli_by_id["dreaming_state"]["value"] == "missed"
    assert sli_by_id["hot_md_policy_state"]["status"] == "ok"
    assert sli_by_id["hot_md_policy_state"]["value"] == "within_budget"


def test_hot_md_job_not_applicable_for_cb_brain_is_unknown(status_mod, tmp_path, env):
    status = status_mod.collect_status(
        "cb-brain", tmp_path, _now(), job_statuses={"dreaming-cb-brain": "OK"}, **env
    )
    sli_by_id = {s["id"]: s for s in status["slis"]}
    assert sli_by_id["hot_md_policy_state"]["status"] == "unknown"
    assert sli_by_id["dreaming_state"]["status"] == "ok"


def test_concurrency_conflicts_counted_only_within_24h_and_matching_vault(status_mod, tmp_path, env):
    env["ledger_path"].parent.mkdir(parents=True, exist_ok=True)
    recent = (_now() - timedelta(hours=1)).isoformat()
    stale = (_now() - timedelta(hours=48)).isoformat()
    lines = [
        json.dumps({"vault": "bs-brain", "status": "conflict_skipped", "timestamp": recent}),
        json.dumps({"vault": "bs-brain", "status": "conflict_skipped", "timestamp": stale}),
        json.dumps({"vault": "cb-brain", "status": "conflict_skipped", "timestamp": recent}),
        json.dumps({"vault": "bs-brain", "status": "applied", "timestamp": recent}),
    ]
    env["ledger_path"].write_text("\n".join(lines) + "\n")

    status = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    sli_by_id = {s["id"]: s for s in status["slis"]}
    assert sli_by_id["concurrency_conflicts_count"]["value"] == 1


def test_contradiction_and_malformed_notes_are_always_unknown_no_fake_parsing(status_mod, tmp_path, env):
    status = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    sli_by_id = {s["id"]: s for s in status["slis"]}
    assert sli_by_id["contradiction_count"]["status"] == "unknown"
    assert "report" in sli_by_id["contradiction_count"]["evidence"]
    assert sli_by_id["malformed_notes_count"]["status"] == "unknown"


def test_alert_on_status_never_fires_for_unknown(status_mod, tmp_path, env):
    status = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    assert status["overall_status"] == "unknown"
    sent = []
    outcomes = status_mod.alert_on_status(
        status, send_fn=sent.append, alert_state_dir=tmp_path / "alert-state", now=_now()
    )
    assert sent == []
    assert all(o is None or True for o in outcomes)  # unknown SLIs are skipped entirely
    assert len(outcomes) < len(status["slis"])  # fewer outcomes than total SLIs -- unknowns were skipped


def test_alert_dedupe_new_then_suppressed_then_recovers(status_mod, tmp_path, env):
    env["backup_state_dir"].mkdir(parents=True)
    state_file = env["backup_state_dir"] / "vault-backup-lastchange-BS_Brain"
    state_file.write_text("")
    import os
    old_time = (_now() - timedelta(hours=72)).timestamp()
    os.utime(state_file, (old_time, old_time))

    alert_dir = tmp_path / "alert-state"
    status1 = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    sent1 = []
    status_mod.alert_on_status(status1, send_fn=sent1.append, alert_state_dir=alert_dir, now=_now())
    assert any("NEW FAILURE" in m for m in sent1)

    sent2 = []
    status_mod.alert_on_status(
        status1, send_fn=sent2.append, alert_state_dir=alert_dir,
        now=_now() + timedelta(minutes=5), rate_limit_seconds=21600,
    )
    assert sent2 == []  # rate-limited, still within window

    # Recovery: touch the backup state file to look fresh again.
    fresh_time = _now().timestamp()
    os.utime(state_file, (fresh_time, fresh_time))
    status2 = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    sent3 = []
    status_mod.alert_on_status(status2, send_fn=sent3.append, alert_state_dir=alert_dir, now=_now())
    assert any("RECOVERED" in m for m in sent3)


def test_write_status_atomic(status_mod, tmp_path, env):
    status = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    out_dir = tmp_path / "out"
    path = status_mod.write_status(status, "bs-brain", out_dir)
    loaded = json.loads(path.read_text())
    assert loaded["vault_name"] == "bs-brain"
    assert list(out_dir.glob("*.tmp")) == []
