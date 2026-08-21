"""Shared alert dedupe/rate-limit primitive for vault observability
(vault-observability-slo build).

Same rate-limiting *idea* as vault-backup.sh's alert_if_due() (a state-file
gates re-alerts), generalized into three distinguishable outcomes instead of
one, and backed by real JSON state (not just an mtime) so a script restart
can tell "this has been failing since X, alerted N times" apart from "first
time seeing this failure" -- the spec's "restarts must not erase recurrent
failure evidence."

State lives under a persistent directory (default
~/.local/state/vault-observability/alert-state/), never /tmp -- mirrors
vault-backup.sh's own STATE_DIR comment about WSL clearing /tmp on reboot.

This module never sends anything itself; callers pass a `send_fn` (e.g. a
Telegram sender) so unit tests can assert on outcomes without any network
dependency.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

DEFAULT_STATE_DIR = Path.home() / ".local" / "state" / "vault-observability" / "alert-state"


class AlertOutcome(str):
    NEW_FAILURE = "new_failure"
    RECURRING_ALERTED = "recurring_alerted"
    RECURRING_SUPPRESSED = "recurring_suppressed"
    RECOVERED = "recovered"
    NO_CHANGE_OK = "no_change_ok"


@dataclass
class IncidentState:
    key: str
    status: str  # "ok" or "failing"
    first_failure_at: str | None
    last_alert_at: str | None
    failure_count_since_recovery: int
    last_message: str

    @classmethod
    def initial(cls, key: str) -> "IncidentState":
        return cls(key=key, status="ok", first_failure_at=None, last_alert_at=None,
                    failure_count_since_recovery=0, last_message="")

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "status": self.status,
            "first_failure_at": self.first_failure_at,
            "last_alert_at": self.last_alert_at,
            "failure_count_since_recovery": self.failure_count_since_recovery,
            "last_message": self.last_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IncidentState":
        return cls(
            key=data["key"],
            status=data.get("status", "ok"),
            first_failure_at=data.get("first_failure_at"),
            last_alert_at=data.get("last_alert_at"),
            failure_count_since_recovery=data.get("failure_count_since_recovery", 0),
            last_message=data.get("last_message", ""),
        )


def _state_path(key: str, state_dir: Path) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return state_dir / f"{safe}.json"


def _load_state(key: str, state_dir: Path) -> IncidentState:
    path = _state_path(key, state_dir)
    try:
        return IncidentState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return IncidentState.initial(key)


def _save_state(state: IncidentState, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = _state_path(state.key, state_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def record_and_maybe_alert(
    key: str,
    is_failing: bool,
    message: str,
    *,
    rate_limit_seconds: int = 3600,
    send_fn: Callable[[str], None] | None = None,
    state_dir: Path = DEFAULT_STATE_DIR,
    now: datetime | None = None,
) -> str:
    """Update persistent incident state for `key` and decide whether to alert.

    Returns one of AlertOutcome's four values. `send_fn`, if given, is called
    with the rendered message exactly when an alert should go out (NEW_FAILURE,
    RECURRING_ALERTED, RECOVERED) -- never for RECURRING_SUPPRESSED or
    NO_CHANGE_OK, so a caller that always passes send_fn gets bounded alert
    volume for free.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    state = _load_state(key, state_dir)

    if not is_failing:
        if state.status == "failing":
            outcome = AlertOutcome.RECOVERED
            rendered = f"RECOVERED: {key} -- {message}"
            if send_fn is not None:
                send_fn(rendered)
            _save_state(IncidentState.initial(key), state_dir)
            return outcome
        return AlertOutcome.NO_CHANGE_OK

    # is_failing == True
    if state.status != "failing":
        state.status = "failing"
        state.first_failure_at = now_iso
        state.failure_count_since_recovery = 1
        state.last_alert_at = now_iso
        state.last_message = message
        _save_state(state, state_dir)
        rendered = f"NEW FAILURE: {key} -- {message}"
        if send_fn is not None:
            send_fn(rendered)
        return AlertOutcome.NEW_FAILURE

    # Already failing -- decide whether the rate limit has elapsed.
    state.failure_count_since_recovery += 1
    state.last_message = message
    last_alert = _parse_iso(state.last_alert_at) if state.last_alert_at else None
    elapsed = (now - last_alert).total_seconds() if last_alert else rate_limit_seconds
    if elapsed >= rate_limit_seconds:
        state.last_alert_at = now_iso
        _save_state(state, state_dir)
        since = state.first_failure_at or now_iso
        rendered = (
            f"STILL FAILING: {key} -- {message} "
            f"(failing since {since}, {state.failure_count_since_recovery} checks)"
        )
        if send_fn is not None:
            send_fn(rendered)
        return AlertOutcome.RECURRING_ALERTED

    _save_state(state, state_dir)
    return AlertOutcome.RECURRING_SUPPRESSED


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def current_state(key: str, state_dir: Path = DEFAULT_STATE_DIR) -> dict:
    """Read-only accessor for the dashboard/aggregator -- never mutates state."""
    return _load_state(key, state_dir).to_dict()
