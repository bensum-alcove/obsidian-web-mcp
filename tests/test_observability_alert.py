"""Tests for obsidian_vault_mcp.observability_alert (vault-observability-slo build)."""

from datetime import datetime, timedelta, timezone

from obsidian_vault_mcp.observability_alert import (
    AlertOutcome,
    current_state,
    record_and_maybe_alert,
)


def _now():
    return datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)


def test_first_failure_alerts_immediately(tmp_path):
    sent = []
    outcome = record_and_maybe_alert(
        "k1", True, "boom", send_fn=sent.append, state_dir=tmp_path, now=_now()
    )
    assert outcome == AlertOutcome.NEW_FAILURE
    assert sent == ["NEW FAILURE: k1 -- boom"]


def test_ok_when_never_failing_does_not_alert(tmp_path):
    sent = []
    outcome = record_and_maybe_alert(
        "k1", False, "fine", send_fn=sent.append, state_dir=tmp_path, now=_now()
    )
    assert outcome == AlertOutcome.NO_CHANGE_OK
    assert sent == []


def test_recurring_failure_within_rate_limit_is_suppressed(tmp_path):
    sent = []
    record_and_maybe_alert("k1", True, "boom", send_fn=sent.append, state_dir=tmp_path, now=_now())
    sent.clear()
    outcome = record_and_maybe_alert(
        "k1", True, "boom", send_fn=sent.append, state_dir=tmp_path,
        now=_now() + timedelta(minutes=5), rate_limit_seconds=3600,
    )
    assert outcome == AlertOutcome.RECURRING_SUPPRESSED
    assert sent == []


def test_recurring_failure_past_rate_limit_re_alerts(tmp_path):
    sent = []
    record_and_maybe_alert("k1", True, "boom", send_fn=sent.append, state_dir=tmp_path, now=_now())
    sent.clear()
    outcome = record_and_maybe_alert(
        "k1", True, "boom", send_fn=sent.append, state_dir=tmp_path,
        now=_now() + timedelta(hours=2), rate_limit_seconds=3600,
    )
    assert outcome == AlertOutcome.RECURRING_ALERTED
    assert len(sent) == 1
    assert "STILL FAILING" in sent[0]
    assert "failing since" in sent[0]


def test_recovery_sends_a_recovered_message_and_clears_state(tmp_path):
    sent = []
    record_and_maybe_alert("k1", True, "boom", send_fn=sent.append, state_dir=tmp_path, now=_now())
    sent.clear()
    outcome = record_and_maybe_alert(
        "k1", False, "boom", send_fn=sent.append, state_dir=tmp_path,
        now=_now() + timedelta(hours=1),
    )
    assert outcome == AlertOutcome.RECOVERED
    assert "RECOVERED: k1" in sent[0]
    assert current_state("k1", state_dir=tmp_path)["status"] == "ok"


def test_state_persists_across_separate_calls_restart_does_not_forget(tmp_path):
    """Simulates a process restart: state is read fresh from disk each call,
    so the failure count/first_failure_at from before a restart survive."""
    record_and_maybe_alert("k1", True, "boom", state_dir=tmp_path, now=_now())
    record_and_maybe_alert("k1", True, "boom", state_dir=tmp_path, now=_now() + timedelta(minutes=1))
    record_and_maybe_alert("k1", True, "boom", state_dir=tmp_path, now=_now() + timedelta(minutes=2))

    state = current_state("k1", state_dir=tmp_path)
    assert state["status"] == "failing"
    assert state["failure_count_since_recovery"] == 3
    assert state["first_failure_at"] is not None


def test_distinct_keys_are_independent(tmp_path):
    sent = []
    record_and_maybe_alert("vault-a:sli", True, "a broke", send_fn=sent.append, state_dir=tmp_path, now=_now())
    record_and_maybe_alert("vault-b:sli", True, "b broke", send_fn=sent.append, state_dir=tmp_path, now=_now())
    assert len(sent) == 2
    assert current_state("vault-a:sli", state_dir=tmp_path)["status"] == "failing"
    assert current_state("vault-b:sli", state_dir=tmp_path)["status"] == "failing"


def test_new_distinct_failure_alerts_even_mid_incident_of_another_key(tmp_path):
    """A storm on one key must never suppress a brand-new, distinct failure."""
    sent = []
    record_and_maybe_alert("k1", True, "first", send_fn=sent.append, state_dir=tmp_path, now=_now())
    outcome = record_and_maybe_alert(
        "k2", True, "second", send_fn=sent.append, state_dir=tmp_path,
        now=_now() + timedelta(seconds=1),
    )
    assert outcome == AlertOutcome.NEW_FAILURE
    assert len(sent) == 2


def test_render_new_failure_hook_overrides_default_wording(tmp_path):
    sent = []
    record_and_maybe_alert(
        "k1", True, "boom", send_fn=sent.append, state_dir=tmp_path, now=_now(),
        render_new_failure=lambda k, m, n: f"CUSTOM DOWN {k}",
    )
    assert sent == ["CUSTOM DOWN k1"]


def test_render_recovered_hook_receives_first_failure_at(tmp_path):
    record_and_maybe_alert("k1", True, "boom", state_dir=tmp_path, now=_now())
    sent = []
    record_and_maybe_alert(
        "k1", False, "boom", send_fn=sent.append, state_dir=tmp_path,
        now=_now() + timedelta(hours=1),
        render_recovered=lambda k, m, ffa, n: f"UP {k} since {ffa}",
    )
    assert sent == [f"UP k1 since {_now().isoformat()}"]


def test_render_recurring_hook_receives_count_and_since(tmp_path):
    record_and_maybe_alert("k1", True, "boom", state_dir=tmp_path, now=_now())
    sent = []
    record_and_maybe_alert(
        "k1", True, "boom", send_fn=sent.append, state_dir=tmp_path,
        now=_now() + timedelta(hours=2), rate_limit_seconds=3600,
        render_recurring=lambda k, m, s, c, n: f"{k} {c} checks since {s}",
    )
    assert sent == [f"k1 2 checks since {_now().isoformat()}"]


def test_no_render_hooks_still_uses_default_templates(tmp_path):
    """Existing callers that never pass render hooks keep exact prior wording."""
    sent = []
    record_and_maybe_alert("k1", True, "boom", send_fn=sent.append, state_dir=tmp_path, now=_now())
    assert sent == ["NEW FAILURE: k1 -- boom"]
