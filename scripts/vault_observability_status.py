#!/usr/bin/env python3
"""vault_observability_status.py — consolidated, machine-readable observability
status for one vault (vault-observability-slo build).

Cron: intended to run every 15 minutes per vault, right after
vault_functional_canary.py (same VAULT_PATH/VAULT_NAME env-var convention as
dreaming.py/job_miss_check.py/vault_functional_canary.py).

Composes ONE JSON snapshot by reading and evaluating (against slo.py's
thresholds) signals that ALREADY EXIST elsewhere in this repo -- it never
recomputes them:

  functional_read_query / index_freshness_seconds  <- vault_functional_canary.py's own status JSON
  mcp_availability                                  <- live probe of this vault's own
                                                        unauthenticated http://127.0.0.1:<port>/health
                                                        at check time (current truth, not history)
  watchdog_recovery_events_24h                      <- check-vault-mcp.sh's watchdog log (restarts, 24h
                                                        window) -- reliability history ONLY, never rendered
                                                        as a current outage (vault-brain-live-health-
                                                        alert-truth-v1, see module tail for why)
  remote_access                                     <- UNKNOWN in production: no non-secret public health
                                                        URL is wired into this repo; see remote_probe param
  backup_age_hours                                  <- vault-backup.sh's per-vault STATE_DIR lastchange file
  restore_drill_age_days                            <- newest ~/backups/vault-clean-room-restore-proof-*/ mtime
  dreaming_state / hot_md_policy_state               <- job_miss_check's own OK/LATE/MISSED classification
  validation_rejects_count                          <- canonical_state_scan.run() (imported directly, read-only)
  concurrency_conflicts_count                        <- dreaming.py's mutation ledger JSONL, 24h window
  retrieval_trend                                    <- evals/history/*.json week-over-week delta (bs-brain only)

Two SLIs (contradiction_count, malformed_notes_count) have no clean
machine-readable source yet -- they are reported Status.UNKNOWN with a
pointer to the report a human should read instead of a fragile ad-hoc
markdown parser. UNKNOWN is a real, visible status (see slo.Status), never
silently folded into "ok".

"Surface machine-readable status on Brain Dashboard/existing monitoring
without making dashboard authoritative": THIS script is the source of
truth. brain-dashboard/main.py's /api/observability/status endpoint only
reads the JSON this writes; it computes nothing itself.

Alerting is layer-grouped and deduplicated via observability_alert.py: one
key per (vault, sli id), so a persisting failure re-alerts on its own
rate-limit schedule instead of spamming every 15-minute run, and restarting
this script never forgets an in-progress incident (state lives on disk).
UNKNOWN never triggers or clears an alert -- a missing data source must not
be misread as either a new failure or a recovery.

vault-brain-live-health-alert-truth-v1 (2026-08-31): mcp_availability used to
equal the 24h watchdog-restart count, so a single restart kept "DOWN" alerting
for a full rolling day even after the service was already healthy again --
confirmed live on 2026-08-31 when all three Brains' /health endpoints returned
200 while Telegram still said "still DOWN ... 99 checks". mcp_availability is
now a live /health probe result only; the restart count moved to its own
watchdog_recovery_events_24h SLI, which is alerted on but can never use DOWN/
RECOVERED wording. Because this changed what "failing" means for the same SLI
id, its alert-state key is versioned (MCP_AVAILABILITY_KEY_SUFFIX below) so a
pre-existing false "failing" incident from the old semantics is never silently
reinterpreted as this new signal's RECOVERED or still-failing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPTS_DIR.parent / "src"
for p in (str(SCRIPTS_DIR), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from obsidian_vault_mcp import config  # noqa: E402
from obsidian_vault_mcp import slo  # noqa: E402
from obsidian_vault_mcp import observability_alert  # noqa: E402
import job_miss_check  # noqa: E402
import job_miss_check_config  # noqa: E402
import canonical_state_scan  # noqa: E402

DEFAULT_STATE_DIR = Path.home() / ".local" / "state" / "vault-observability"

# Australia/Brisbane, fixed UTC+10, no DST -- this box's cron/health-check
# convention throughout the repo (see check-vault-mcp.sh watchdog log
# timestamps, health-check.sh comments). Alert messages render in this zone
# so "checked HH:MM AEST" matches what Ben sees on his clock.
AEST = timezone(timedelta(hours=10))

VAULT_DISPLAY_NAMES = {
    "bs-brain": "BS Brain",
    "cb-brain": "CB Brain",
    "alcove-brain": "Alcove Brain",
}
DEFAULT_BACKUP_STATE_DIR = Path(
    os.environ.get("VAULT_BACKUP_STATE_DIR", str(Path.home() / ".local" / "state" / "vault-backup"))
)
DEFAULT_BACKUPS_ROOT = Path.home() / "backups"
MUTATION_LEDGER_PATH = Path.home() / ".build-orchestrator" / "ledgers" / "dreaming-mutations.jsonl"
EVALS_HISTORY_DIR = SCRIPTS_DIR.parent / "evals" / "history"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "8558481275")

# Per-vault wiring for signals whose location differs by vault. Values taken
# directly from the live crontab / vault-backup.sh / check-vault-mcp.sh
# invocations at build time -- see vault-observability-slo-output.md.
VAULT_PROFILES = {
    "bs-brain": {
        "backup_state_name": "BS_Brain",
        "watchdog_log": Path("/tmp/vault-mcp-watchdog.log"),
        "dreaming_job": "dreaming-bs-brain",
        "hot_md_job": "hot-md-curate",
        "canonical_state_scan": True,
        "retrieval_eval": True,
        "health_port": 8420,
    },
    "cb-brain": {
        "backup_state_name": "CB_Brain",
        "watchdog_log": Path("/tmp/cb-vault-mcp-watchdog.log"),
        "dreaming_job": "dreaming-cb-brain",
        "hot_md_job": None,
        "canonical_state_scan": False,
        "retrieval_eval": False,
        "health_port": 8423,
    },
    "alcove-brain": {
        "backup_state_name": "Alcove_Brain",
        "watchdog_log": Path("/tmp/alcove-vault-mcp-watchdog.log"),
        "dreaming_job": "dreaming-alcove-brain",
        "hot_md_job": None,
        "canonical_state_scan": False,
        "retrieval_eval": False,
        "health_port": 8426,
    },
}

# mcp_availability's meaning changed (2026-08-31, vault-brain-live-health-alert-
# truth-v1) from a 24h watchdog-restart count to a live /health probe result --
# see module docstring. Suffixing the alert-state key stops a pre-existing
# "failing" incident under the old semantics from being silently read as this
# new signal recovering or still failing; the old unsuffixed state file is left
# on disk untouched, for audit.
MCP_AVAILABILITY_KEY_SUFFIX = ":live-health-v1"


def _hours_since(mtime: float, now: datetime) -> float:
    return (now.timestamp() - mtime) / 3600.0


def read_canary_status(vault_name: str, status_dir: Path) -> dict | None:
    path = status_dir / f"canary-{vault_name}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def probe_local_health(port: int, timeout: float = 5.0) -> str:
    """Live current-health probe of this vault's own unauthenticated /health
    endpoint (server.py's `/health` route, always {"status": "ok"} on 200).
    Returns "up" for HTTP 200, "down" for anything else -- timeout, connection
    refused, non-200 -- these must all read as one current-origin-failure
    class to the caller, never distinguished by cause here."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as resp:
            return "up" if resp.status == 200 else "down"
    except Exception:
        return "down"


def read_watchdog_restarts_24h(log_path: Path, now: datetime) -> int | None:
    """Count '[WATCHDOG] ... Forcing recovery' restart lines in the last 24h."""
    if not log_path.exists():
        return None
    count = 0
    cutoff = now.timestamp() - 86400
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "[WATCHDOG]" not in line or "Forcing recovery" not in line:
                    continue
                ts_str = line[:19]  # "YYYY-MM-DD HH:MM:SS"
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(
                        tzinfo=timezone.utc
                    ).timestamp()
                except ValueError:
                    continue
                if ts >= cutoff:
                    count += 1
    except OSError:
        return None
    return count


def read_backup_age_hours(vault_backup_name: str, state_dir: Path, now: datetime) -> float | None:
    path = state_dir / f"vault-backup-lastchange-{vault_backup_name}"
    try:
        return _hours_since(path.stat().st_mtime, now)
    except FileNotFoundError:
        return None


def read_restore_drill_age_days(backups_root: Path, now: datetime) -> float | None:
    candidates = sorted(backups_root.glob("vault-clean-room-restore-proof-*"))
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return _hours_since(newest.stat().st_mtime, now) / 24.0


def job_miss_statuses(now: datetime, jobs: list[dict] | None = None) -> dict[str, str]:
    """Thin wrapper around job_miss_check.check_jobs -- takes an explicit
    `jobs` list (defaulting to the real job_miss_check_config.JOBS) so tests
    can inject fixture jobs instead of reading this box's real artifacts."""
    results = job_miss_check.check_jobs(
        jobs if jobs is not None else job_miss_check_config.JOBS, now
    )
    return {r["name"]: r["status"] for r in results}


def read_job_state(job_name: str | None, statuses: dict[str, str]) -> str | None:
    if job_name is None:
        return None
    status = statuses.get(job_name)
    if status is None:
        return None
    return "completed" if status == "OK" else "missed"


def read_validation_rejects(vault_path: Path) -> int | None:
    records_dir = (
        vault_path / "BS 2nd Brain" / "Alcove" / "Infrastructure" / "Canonical State" / "records"
    )
    if not records_dir.is_dir():
        return None
    report = canonical_state_scan.run(records_dir)
    return len(report["malformed_records"]) + len(report["duplicate_authority"])


def read_concurrency_conflicts_24h(vault_name: str, ledger_path: Path, now: datetime) -> int | None:
    if not ledger_path.exists():
        return None
    cutoff = now.timestamp() - 86400
    count = 0
    try:
        with open(ledger_path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("vault") != vault_name or record.get("status") != "conflict_skipped":
                    continue
                try:
                    ts = datetime.fromisoformat(record["timestamp"]).timestamp()
                except (KeyError, ValueError):
                    continue
                if ts >= cutoff:
                    count += 1
    except OSError:
        return None
    return count


def read_retrieval_trend(history_dir: Path, tool: str = "vault_search") -> float | None:
    files = sorted(history_dir.glob("*.json"))
    if len(files) < 2:
        return None
    try:
        latest = json.loads(files[-1].read_text(encoding="utf-8"))
        previous = json.loads(files[-2].read_text(encoding="utf-8"))
        return latest[tool]["overall"]["r_at_5"] - previous[tool]["overall"]["r_at_5"]
    except (KeyError, json.JSONDecodeError):
        return None


def collect_status(
    vault_name: str,
    vault_path: Path,
    now: datetime,
    *,
    status_dir: Path = DEFAULT_STATE_DIR,
    backup_state_dir: Path = DEFAULT_BACKUP_STATE_DIR,
    backups_root: Path = DEFAULT_BACKUPS_ROOT,
    ledger_path: Path = MUTATION_LEDGER_PATH,
    history_dir: Path = EVALS_HISTORY_DIR,
    watchdog_log: Path | None = None,
    job_statuses: dict[str, str] | None = None,
    health_probe: Callable[[], str] | None = None,
    remote_probe: Callable[[], str] | None = None,
) -> dict:
    profile = dict(VAULT_PROFILES.get(vault_name, {}))
    if watchdog_log is not None:
        profile["watchdog_log"] = watchdog_log
    if job_statuses is None:
        job_statuses = job_miss_statuses(now)
    canary = read_canary_status(vault_name, status_dir)

    values: dict[str, float | str | None] = {}
    evidence: dict[str, str] = {}

    if canary is not None:
        values["functional_read_query"] = len(canary.get("layers_failing", []))
        index_layer = next(
            (l for l in canary.get("layers", []) if l["layer"] == "verify_index_sees_patch"), None
        )
        if index_layer is not None and index_layer["ok"]:
            checked_at = datetime.fromisoformat(canary["checked_at"])
            values["index_freshness_seconds"] = (now - checked_at).total_seconds()
        else:
            values["index_freshness_seconds"] = None
        evidence["functional_read_query"] = str(status_dir / f"canary-{vault_name}.json")
        evidence["index_freshness_seconds"] = evidence["functional_read_query"]
    else:
        values["functional_read_query"] = None
        values["index_freshness_seconds"] = None
        evidence["functional_read_query"] = "no canary status file found -- has it ever run?"
        evidence["index_freshness_seconds"] = evidence["functional_read_query"]

    health_port = profile.get("health_port")
    if health_probe is not None:
        values["mcp_availability"] = health_probe()
    elif health_port is not None:
        values["mcp_availability"] = probe_local_health(health_port)
    else:
        values["mcp_availability"] = None
    evidence["mcp_availability"] = (
        f"live probe http://127.0.0.1:{health_port}/health" if health_port is not None
        else "no local health port configured for this vault"
    )

    values["remote_access"] = remote_probe() if remote_probe is not None else None
    evidence["remote_access"] = (
        "remote_probe callable result (test double)" if remote_probe is not None
        else "no non-secret, unauthenticated public health URL is wired for this "
             "vault -- remote path is not tracked; mcp_availability's local /health "
             "probe above remains the sole origin-truth signal"
    )

    watchdog_log = profile.get("watchdog_log")
    values["watchdog_recovery_events_24h"] = (
        read_watchdog_restarts_24h(watchdog_log, now) if watchdog_log else None
    )
    evidence["watchdog_recovery_events_24h"] = (
        str(watchdog_log) if watchdog_log else "no watchdog log configured"
    )

    backup_name = profile.get("backup_state_name")
    values["backup_age_hours"] = (
        read_backup_age_hours(backup_name, backup_state_dir, now) if backup_name else None
    )
    evidence["backup_age_hours"] = str(backup_state_dir / f"vault-backup-lastchange-{backup_name}") \
        if backup_name else "no backup state name configured"

    values["restore_drill_age_days"] = read_restore_drill_age_days(backups_root, now)
    evidence["restore_drill_age_days"] = str(backups_root / "vault-clean-room-restore-proof-*")

    values["dreaming_state"] = read_job_state(profile.get("dreaming_job"), job_statuses)
    evidence["dreaming_state"] = f"job_miss_check job={profile.get('dreaming_job')!r}"

    _hot_md_job_state = read_job_state(profile.get("hot_md_job"), job_statuses)
    values["hot_md_policy_state"] = (
        {"completed": "within_budget", "missed": "over_budget"}.get(_hot_md_job_state)
    )
    evidence["hot_md_policy_state"] = f"job_miss_check job={profile.get('hot_md_job')!r}"

    values["validation_rejects_count"] = (
        read_validation_rejects(vault_path) if profile.get("canonical_state_scan") else None
    )
    evidence["validation_rejects_count"] = (
        "scripts/canonical_state_scan.py" if profile.get("canonical_state_scan")
        else "canonical-state scanning is BS-Brain-specific infrastructure; not applicable here"
    )

    values["concurrency_conflicts_count"] = read_concurrency_conflicts_24h(vault_name, ledger_path, now)
    evidence["concurrency_conflicts_count"] = str(ledger_path)

    values["retrieval_trend"] = (
        read_retrieval_trend(history_dir) if profile.get("retrieval_eval") else None
    )
    evidence["retrieval_trend"] = (
        str(history_dir) if profile.get("retrieval_eval")
        else "vault-retrieval-eval only runs for bs-brain currently"
    )

    # No clean machine-readable source yet -- report UNKNOWN, not "ok".
    values["contradiction_count"] = None
    evidence["contradiction_count"] = (
        "BS 2nd Brain/Alcove/Infrastructure/contradiction-lint-report.md "
        "(no machine-readable API yet -- read the report)"
    )
    values["malformed_notes_count"] = None
    evidence["malformed_notes_count"] = (
        "dreaming report (BS 2nd Brain/Alcove/Infrastructure/dreaming-reports/*.md "
        "or _Reports/dreaming/*.md) -- no machine-readable API yet"
    )

    slis_out = []
    layer_status: dict[str, list[str]] = {}
    for sli_id, sli in slo.REGISTRY.items():
        value = values.get(sli_id)
        status = sli.evaluate(value)
        slis_out.append({
            "id": sli_id,
            "layer": sli.layer,
            "description": sli.description,
            "unit": sli.unit,
            "value": value,
            "status": status.value,
            "warning": sli.warning,
            "critical": sli.critical,
            "owner": sli.owner,
            "runbook": sli.runbook,
            "evidence": evidence.get(sli_id, ""),
        })
        layer_status.setdefault(sli.layer, []).append(status.value)

    severity_rank = {"unknown": 0, "ok": 1, "warning": 2, "critical": 3}
    layers_out = {
        layer: max(statuses, key=lambda s: severity_rank[s])
        for layer, statuses in layer_status.items()
    }
    overall = max((s["status"] for s in slis_out), key=lambda s: severity_rank[s], default="unknown")

    return {
        "vault_name": vault_name,
        "checked_at": now.isoformat(),
        "generated_by": "vault_observability_status.py",
        "overall_status": overall,
        "layers": layers_out,
        "slis": slis_out,
    }


def _aest_hhmm(dt: datetime) -> str:
    return dt.astimezone(AEST).strftime("%H:%M")


def _availability_renderers(display_name: str):
    """Clear DOWN/RECOVERED wording for the mcp_availability SLI -- this is
    the signal that actually means "is this Brain's MCP reachable", so it
    gets human phrasing (explicit DOWN/RECOVERED + AEST timestamp + service
    name) instead of the generic "[STATUS] sli_id = value" template other
    SLIs use, since those aren't about service downtime and "DOWN" would be
    misleading applied to e.g. backup_age_hours.
    """

    def render_new_failure(key: str, message: str, now: datetime) -> str:
        return f"{display_name} DOWN — checked {_aest_hhmm(now)} AEST"

    def render_recurring(key: str, message: str, first_failure_at: str, failure_count: int, now: datetime) -> str:
        since = _aest_hhmm(datetime.fromisoformat(first_failure_at)) if first_failure_at else "unknown"
        return (
            f"{display_name} still DOWN — checked {_aest_hhmm(now)} AEST "
            f"(down since {since} AEST, {failure_count} checks)"
        )

    def render_recovered(key: str, message: str, first_failure_at: str | None, now: datetime) -> str:
        if not first_failure_at:
            return f"{display_name} RECOVERED — checked {_aest_hhmm(now)} AEST"
        since_dt = datetime.fromisoformat(first_failure_at)
        minutes = max(0, int((now - since_dt).total_seconds() // 60))
        duration = f"{minutes}m" if minutes < 60 else f"{minutes // 60}h{minutes % 60:02d}m"
        return (
            f"{display_name} RECOVERED — checked {_aest_hhmm(now)} AEST "
            f"(down for {duration}, since {_aest_hhmm(since_dt)} AEST)"
        )

    return render_new_failure, render_recurring, render_recovered


def _watchdog_renderers(display_name: str):
    """Reliability-history wording for watchdog_recovery_events_24h -- a restart
    count over a rolling 24h window, NOT current availability. Must never say
    DOWN, still DOWN, down since, or RECOVERED (vault-brain-live-health-alert-
    truth-v1) so it can never be misread as a current outage."""

    def render_new_failure(key: str, message: str, now: datetime) -> str:
        return (
            f"{display_name} watchdog recovery events elevated — checked {_aest_hhmm(now)} AEST "
            f"(reliability history, not a current outage)"
        )

    def render_recurring(key: str, message: str, first_failure_at: str, failure_count: int, now: datetime) -> str:
        since = _aest_hhmm(datetime.fromisoformat(first_failure_at)) if first_failure_at else "unknown"
        return (
            f"{display_name} watchdog recovery events still elevated — checked {_aest_hhmm(now)} AEST "
            f"(elevated since {since} AEST, {failure_count} checks; reliability history, not a current outage)"
        )

    def render_resolved(key: str, message: str, first_failure_at: str | None, now: datetime) -> str:
        return f"{display_name} watchdog recovery events back to baseline — checked {_aest_hhmm(now)} AEST"

    return render_new_failure, render_recurring, render_resolved


def _remote_access_renderers(display_name: str):
    """Remote-path wording for remote_access -- must never say "<Brain> DOWN"
    when local origin health is fine (vault-brain-live-health-alert-truth-v1)."""

    def render_new_failure(key: str, message: str, now: datetime) -> str:
        return f"{display_name} local healthy — remote access degraded, checked {_aest_hhmm(now)} AEST"

    def render_recurring(key: str, message: str, first_failure_at: str, failure_count: int, now: datetime) -> str:
        since = _aest_hhmm(datetime.fromisoformat(first_failure_at)) if first_failure_at else "unknown"
        return (
            f"{display_name} local healthy — remote access still degraded, checked {_aest_hhmm(now)} AEST "
            f"(degraded since {since} AEST, {failure_count} checks)"
        )

    def render_recovered(key: str, message: str, first_failure_at: str | None, now: datetime) -> str:
        return f"{display_name} remote access restored — checked {_aest_hhmm(now)} AEST"

    return render_new_failure, render_recurring, render_recovered


def send_telegram(message: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        return
    payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message}).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except Exception as exc:
        print(f"Telegram send failed: {exc}", file=sys.stderr)


def alert_on_status(
    status: dict,
    *,
    rate_limit_seconds: int = 21600,
    send_fn=send_telegram,
    alert_state_dir: Path = observability_alert.DEFAULT_STATE_DIR,
    now: datetime | None = None,
) -> list[str]:
    """Dedupe/rate-limit alerting per SLI. UNKNOWN never alerts or clears an
    incident -- see module docstring. Returns the list of alert outcomes for
    logging/testing."""
    outcomes = []
    display_name = VAULT_DISPLAY_NAMES.get(status["vault_name"], status["vault_name"])
    for sli_status in status["slis"]:
        if sli_status["status"] == "unknown":
            continue
        key_suffix = MCP_AVAILABILITY_KEY_SUFFIX if sli_status["id"] == "mcp_availability" else ""
        key = f"{status['vault_name']}:{sli_status['id']}{key_suffix}"
        is_failing = sli_status["status"] in ("warning", "critical")
        message = (
            f"[{sli_status['status'].upper()}] {sli_status['id']} = {sli_status['value']} "
            f"{sli_status['unit']} (owner: {sli_status['owner']}). {sli_status['runbook']}"
        )
        renderers = {}
        if sli_status["id"] == "mcp_availability":
            render_new_failure, render_recurring, render_recovered = _availability_renderers(display_name)
            renderers = dict(
                render_new_failure=render_new_failure,
                render_recurring=render_recurring,
                render_recovered=render_recovered,
            )
        elif sli_status["id"] == "watchdog_recovery_events_24h":
            render_new_failure, render_recurring, render_recovered = _watchdog_renderers(display_name)
            renderers = dict(
                render_new_failure=render_new_failure,
                render_recurring=render_recurring,
                render_recovered=render_recovered,
            )
        elif sli_status["id"] == "remote_access":
            render_new_failure, render_recurring, render_recovered = _remote_access_renderers(display_name)
            renderers = dict(
                render_new_failure=render_new_failure,
                render_recurring=render_recurring,
                render_recovered=render_recovered,
            )
        outcome = observability_alert.record_and_maybe_alert(
            key, is_failing, message,
            rate_limit_seconds=rate_limit_seconds, send_fn=send_fn,
            state_dir=alert_state_dir, now=now, **renderers,
        )
        outcomes.append(outcome)
    return outcomes


def write_status(status: dict, vault_name: str, status_dir: Path) -> Path:
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / f"status-{vault_name}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault-path", type=Path, default=config.VAULT_PATH)
    parser.add_argument("--vault-name", default=os.environ.get("VAULT_NAME", config.VAULT_PATH.name))
    parser.add_argument("--status-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--no-alert", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    status = collect_status(args.vault_name, args.vault_path, now, status_dir=args.status_dir)
    out_path = write_status(status, args.vault_name, args.status_dir)

    if not args.no_alert:
        alert_on_status(status, now=now)

    print(json.dumps(status, indent=2))
    print(f"status written to {out_path}", file=sys.stderr)
    return 0 if status["overall_status"] not in ("critical",) else 1


if __name__ == "__main__":
    raise SystemExit(main())
