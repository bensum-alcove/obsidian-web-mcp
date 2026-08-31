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
        # Hermetic default: no test should hit the real network via
        # probe_local_health unless it explicitly opts in. Tests that care
        # about "no health data available" override this back to `lambda: None`.
        "health_probe": lambda: "up",
    }


def test_all_sources_missing_is_unknown_not_ok(status_mod, tmp_path, env):
    """No data anywhere must never be silently read as healthy."""
    env = dict(env, health_probe=lambda: None)
    status = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    sli_by_id = {s["id"]: s for s in status["slis"]}
    for sli_id in ("functional_read_query", "backup_age_hours", "restore_drill_age_days", "mcp_availability"):
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
    env = dict(env, health_probe=lambda: None)
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


def _mcp_status(vault_name: str, sli_status: str, value="down") -> dict:
    return {
        "vault_name": vault_name,
        "slis": [{
            "id": "mcp_availability",
            "status": sli_status,
            "value": value,
            "unit": "probe_result",
            "owner": "ben",
            "runbook": "Check supervisord status and this vault's process/port -- a live "
                       "/health probe just failed or timed out at check time.",
        }],
    }


def test_mcp_availability_new_failure_says_down_with_display_name_and_aest(status_mod, tmp_path):
    status = _mcp_status("bs-brain", "critical")
    sent = []
    status_mod.alert_on_status(
        status, send_fn=sent.append, alert_state_dir=tmp_path / "alert-state", now=_now(),
    )
    assert sent == ["BS Brain DOWN — checked 22:00 AEST"]  # _now() 12:00 UTC == 22:00 AEST


def test_mcp_availability_recurring_failure_says_still_down_with_since(status_mod, tmp_path):
    alert_dir = tmp_path / "alert-state"
    status = _mcp_status("bs-brain", "critical")
    status_mod.alert_on_status(status, send_fn=lambda m: None, alert_state_dir=alert_dir, now=_now())

    sent = []
    status_mod.alert_on_status(
        status, send_fn=sent.append, alert_state_dir=alert_dir,
        now=_now() + timedelta(hours=7), rate_limit_seconds=21600,
    )
    assert len(sent) == 1
    assert "still DOWN" in sent[0]
    assert "down since 22:00 AEST" in sent[0]
    assert "checked 05:00 AEST" in sent[0]


def test_mcp_availability_recurring_failure_within_window_is_suppressed(status_mod, tmp_path):
    alert_dir = tmp_path / "alert-state"
    status = _mcp_status("bs-brain", "critical")
    status_mod.alert_on_status(status, send_fn=lambda m: None, alert_state_dir=alert_dir, now=_now())

    sent = []
    status_mod.alert_on_status(
        status, send_fn=sent.append, alert_state_dir=alert_dir,
        now=_now() + timedelta(minutes=5), rate_limit_seconds=21600,
    )
    assert sent == []


def test_mcp_availability_recovery_says_recovered_with_duration(status_mod, tmp_path):
    alert_dir = tmp_path / "alert-state"
    status_mod.alert_on_status(
        _mcp_status("bs-brain", "critical"), send_fn=lambda m: None, alert_state_dir=alert_dir, now=_now(),
    )

    sent = []
    status_mod.alert_on_status(
        _mcp_status("bs-brain", "ok", value="up"), send_fn=sent.append, alert_state_dir=alert_dir,
        now=_now() + timedelta(minutes=45),
    )
    assert len(sent) == 1
    assert sent[0].startswith("BS Brain RECOVERED")
    assert "down for 45m" in sent[0]


def test_mcp_availability_new_outage_after_recovery_alerts_again(status_mod, tmp_path):
    alert_dir = tmp_path / "alert-state"
    down_status = _mcp_status("bs-brain", "critical")
    up_status = _mcp_status("bs-brain", "ok", value="up")

    status_mod.alert_on_status(down_status, send_fn=lambda m: None, alert_state_dir=alert_dir, now=_now())
    status_mod.alert_on_status(
        up_status, send_fn=lambda m: None, alert_state_dir=alert_dir, now=_now() + timedelta(hours=1),
    )

    sent = []
    status_mod.alert_on_status(
        down_status, send_fn=sent.append, alert_state_dir=alert_dir, now=_now() + timedelta(hours=2),
    )
    assert len(sent) == 1
    assert sent[0].startswith("BS Brain DOWN")


def test_mcp_availability_unknown_does_not_alert_or_clear_in_progress_incident(status_mod, tmp_path):
    from obsidian_vault_mcp.observability_alert import current_state

    alert_dir = tmp_path / "alert-state"
    status_mod.alert_on_status(
        _mcp_status("bs-brain", "critical"), send_fn=lambda m: None, alert_state_dir=alert_dir, now=_now(),
    )

    sent = []
    status_mod.alert_on_status(
        _mcp_status("bs-brain", "unknown", value=None), send_fn=sent.append, alert_state_dir=alert_dir,
        now=_now() + timedelta(minutes=10),
    )
    assert sent == []
    assert current_state(
        f"bs-brain:mcp_availability{status_mod.MCP_AVAILABILITY_KEY_SUFFIX}", state_dir=alert_dir
    )["status"] == "failing"


def test_non_availability_slis_keep_generic_wording_not_down(status_mod, tmp_path):
    """DOWN/RECOVERED phrasing is specific to mcp_availability -- it would be
    misleading applied to e.g. a stale backup, which isn't a service outage."""
    status = {
        "vault_name": "bs-brain",
        "slis": [{
            "id": "backup_age_hours", "status": "critical", "value": 72,
            "unit": "hours", "owner": "ben", "runbook": "run vault-backup.sh",
        }],
    }
    sent = []
    status_mod.alert_on_status(status, send_fn=sent.append, alert_state_dir=tmp_path / "alert-state", now=_now())
    assert len(sent) == 1
    assert "DOWN" not in sent[0]
    assert sent[0].startswith("NEW FAILURE:")


# --- vault-brain-live-health-alert-truth-v1 regression coverage -----------
# Reproduces and pins closed the 2026-08-31 false-positive: three healthy
# Brains kept alerting "still DOWN" because mcp_availability was a 24h
# watchdog-restart count, not a live probe.

def test_historical_watchdog_event_plus_current_200_is_no_down(status_mod, tmp_path, env):
    """The exact 2026-08-31 scenario: a restart happened recently (so the
    watchdog log has a 'Forcing recovery' line inside the 24h window) but the
    live /health probe is healthy right now -- must never say DOWN."""
    env["watchdog_log"].write_text(
        "2026-08-30 19:34:01 [WATCHDOG] bs-brain-vault unhealthy — HTTP 000 (expected 401). "
        "Forcing recovery.\n"
    )
    env = dict(env, health_probe=lambda: "up")
    status = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    sli_by_id = {s["id"]: s for s in status["slis"]}
    assert sli_by_id["mcp_availability"]["value"] == "up"
    assert sli_by_id["mcp_availability"]["status"] == "ok"
    assert sli_by_id["watchdog_recovery_events_24h"]["value"] == 1

    sent = []
    status_mod.alert_on_status(status, send_fn=sent.append, alert_state_dir=tmp_path / "alert-state", now=_now())
    assert not any("DOWN" in m for m in sent)


def test_three_healthy_brains_with_stale_old_alert_state_is_no_still_down(status_mod, tmp_path):
    """A pre-existing 'failing' incident under the OLD unversioned key (the
    real production state found on 2026-08-31: 100 checks, status=failing)
    must not bleed into the new live-health signal as a still-DOWN or a fake
    RECOVERED -- the versioned key starts fresh and old state is left alone."""
    from obsidian_vault_mcp.observability_alert import current_state

    alert_dir = tmp_path / "alert-state"
    alert_dir.mkdir(parents=True)
    old_key_path = alert_dir / "bs-brain_mcp_availability.json"
    old_key_path.write_text(json.dumps({
        "key": "bs-brain:mcp_availability",
        "status": "failing",
        "first_failure_at": "2026-08-30T09:45:01.371972+00:00",
        "last_alert_at": "2026-08-31T10:15:01.854886+00:00",
        "failure_count_since_recovery": 100,
        "last_message": "stale pre-migration state",
    }))

    sent = []
    status_mod.alert_on_status(
        _mcp_status("bs-brain", "ok", value="up"), send_fn=sent.append, alert_state_dir=alert_dir, now=_now(),
    )
    assert sent == []  # no still-DOWN, no fake RECOVERED
    assert json.loads(old_key_path.read_text())["failure_count_since_recovery"] == 100  # untouched, preserved for audit

    new_key = f"bs-brain:mcp_availability{status_mod.MCP_AVAILABILITY_KEY_SUFFIX}"
    assert current_state(new_key, state_dir=alert_dir)["status"] == "ok"


def test_consecutive_current_health_failures_have_truthful_count(status_mod, tmp_path):
    """'N checks' must count consecutive current-probe failures only, never
    an inflated historical restart count."""
    alert_dir = tmp_path / "alert-state"
    down = _mcp_status("bs-brain", "critical")
    status_mod.alert_on_status(down, send_fn=lambda m: None, alert_state_dir=alert_dir, now=_now())
    status_mod.alert_on_status(
        down, send_fn=lambda m: None, alert_state_dir=alert_dir,
        now=_now() + timedelta(hours=1), rate_limit_seconds=3600,
    )
    sent = []
    status_mod.alert_on_status(
        down, send_fn=sent.append, alert_state_dir=alert_dir,
        now=_now() + timedelta(hours=2), rate_limit_seconds=3600,
    )
    assert len(sent) == 1
    assert "3 checks" in sent[0]


def test_current_recovery_produces_exactly_one_recovered_message(status_mod, tmp_path):
    alert_dir = tmp_path / "alert-state"
    sent = []
    status_mod.alert_on_status(
        _mcp_status("bs-brain", "critical"), send_fn=sent.append, alert_state_dir=alert_dir, now=_now(),
    )
    status_mod.alert_on_status(
        _mcp_status("bs-brain", "ok", value="up"), send_fn=sent.append, alert_state_dir=alert_dir,
        now=_now() + timedelta(minutes=30),
    )
    recovered = [m for m in sent if "RECOVERED" in m]
    assert len(recovered) == 1
    assert recovered[0].startswith("BS Brain RECOVERED")


def test_local_healthy_remote_unhealthy_uses_remote_wording_not_brain_down(status_mod, tmp_path, env):
    env = dict(env, health_probe=lambda: "up", remote_probe=lambda: "down")
    status = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    sli_by_id = {s["id"]: s for s in status["slis"]}
    assert sli_by_id["mcp_availability"]["status"] == "ok"
    assert sli_by_id["remote_access"]["status"] == "critical"

    sent = []
    status_mod.alert_on_status(status, send_fn=sent.append, alert_state_dir=tmp_path / "alert-state", now=_now())
    remote_messages = [m for m in sent if "remote access" in m]
    assert len(remote_messages) == 1
    assert "DOWN" not in remote_messages[0]
    assert "degraded" in remote_messages[0]
    assert not any(m.startswith("BS Brain DOWN") for m in sent)


def test_remote_access_is_unknown_by_default_no_probe_wired(status_mod, tmp_path, env):
    """No remote_probe configured (the real production default -- no safe
    secret-free URL exists) must report unknown, never down."""
    status = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    sli_by_id = {s["id"]: s for s in status["slis"]}
    assert sli_by_id["remote_access"]["status"] == "unknown"

    sent = []
    status_mod.alert_on_status(status, send_fn=sent.append, alert_state_dir=tmp_path / "alert-state", now=_now())
    assert not any("remote access" in m for m in sent)


def test_current_local_failure_with_zero_watchdog_restarts_still_says_down(status_mod, tmp_path, env):
    """DOWN must be driven purely by the live probe -- a genuine current
    failure with a clean (zero-restart) watchdog history must still alert."""
    env["watchdog_log"].write_text("")  # exists, but no "Forcing recovery" lines -- a real zero, not unknown
    env = dict(env, health_probe=lambda: "down")
    status = status_mod.collect_status("bs-brain", tmp_path, _now(), job_statuses={}, **env)
    sli_by_id = {s["id"]: s for s in status["slis"]}
    assert sli_by_id["mcp_availability"]["status"] == "critical"
    assert sli_by_id["watchdog_recovery_events_24h"]["value"] == 0

    sent = []
    status_mod.alert_on_status(status, send_fn=sent.append, alert_state_dir=tmp_path / "alert-state", now=_now())
    assert any(m.startswith("BS Brain DOWN") for m in sent)


def test_watchdog_recovery_events_never_use_down_or_recovered_wording(status_mod, tmp_path):
    """Elevated-then-resolved watchdog events must never render with DOWN/
    still DOWN/down since/RECOVERED wording -- that vocabulary is reserved for
    mcp_availability's genuine current-health incidents."""
    def _watchdog_status(vault_name, sli_status, value=2):
        return {
            "vault_name": vault_name,
            "slis": [{
                "id": "watchdog_recovery_events_24h",
                "status": sli_status,
                "value": value,
                "unit": "restarts_per_24h",
                "owner": "ben",
                "runbook": "check-vault-mcp.sh already self-heals.",
            }],
        }

    alert_dir = tmp_path / "alert-state"
    forbidden = ("DOWN", "RECOVERED")

    sent = []
    status_mod.alert_on_status(
        _watchdog_status("bs-brain", "warning"), send_fn=sent.append, alert_state_dir=alert_dir, now=_now(),
    )
    status_mod.alert_on_status(
        _watchdog_status("bs-brain", "ok", value=0), send_fn=sent.append, alert_state_dir=alert_dir,
        now=_now() + timedelta(minutes=30),
    )
    assert len(sent) == 2
    for message in sent:
        for word in forbidden:
            assert word not in message, message
