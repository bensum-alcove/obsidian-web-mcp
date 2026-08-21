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
  mcp_availability                                  <- check-vault-mcp.sh's watchdog log (restarts, 24h window)
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
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

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
    },
    "cb-brain": {
        "backup_state_name": "CB_Brain",
        "watchdog_log": Path("/tmp/cb-vault-mcp-watchdog.log"),
        "dreaming_job": "dreaming-cb-brain",
        "hot_md_job": None,
        "canonical_state_scan": False,
        "retrieval_eval": False,
    },
    "alcove-brain": {
        "backup_state_name": "Alcove_Brain",
        "watchdog_log": Path("/tmp/alcove-vault-mcp-watchdog.log"),
        "dreaming_job": "dreaming-alcove-brain",
        "hot_md_job": None,
        "canonical_state_scan": False,
        "retrieval_eval": False,
    },
}


def _hours_since(mtime: float, now: datetime) -> float:
    return (now.timestamp() - mtime) / 3600.0


def read_canary_status(vault_name: str, status_dir: Path) -> dict | None:
    path = status_dir / f"canary-{vault_name}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


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

    watchdog_log = profile.get("watchdog_log")
    values["mcp_availability"] = (
        read_watchdog_restarts_24h(watchdog_log, now) if watchdog_log else None
    )
    evidence["mcp_availability"] = str(watchdog_log) if watchdog_log else "no watchdog log configured"

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
    for sli_status in status["slis"]:
        if sli_status["status"] == "unknown":
            continue
        key = f"{status['vault_name']}:{sli_status['id']}"
        is_failing = sli_status["status"] in ("warning", "critical")
        message = (
            f"[{sli_status['status'].upper()}] {sli_status['id']} = {sli_status['value']} "
            f"{sli_status['unit']} (owner: {sli_status['owner']}). {sli_status['runbook']}"
        )
        outcome = observability_alert.record_and_maybe_alert(
            key, is_failing, message,
            rate_limit_seconds=rate_limit_seconds, send_fn=send_fn,
            state_dir=alert_state_dir, now=now,
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
        alert_on_status(status)

    print(json.dumps(status, indent=2))
    print(f"status written to {out_path}", file=sys.stderr)
    return 0 if status["overall_status"] not in ("critical",) else 1


if __name__ == "__main__":
    raise SystemExit(main())
