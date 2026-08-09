#!/usr/bin/env python3
"""job_miss_check.py — Alert when a scheduled job silently fails to run.

Cron: 0 10 * * * (10:00 AEST daily). Read-only observer: for each job in
job_miss_check_config.JOBS, compares the job's expected artifact against how
fresh it should be and sends one consolidated Telegram alert listing anything
not OK. Sends nothing when everything is OK — silence means healthy.

This script never touches, re-triggers, or repairs a monitored job. It only
reads mtimes.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from job_miss_check_config import JOBS  # noqa: E402

LOG_PATH = Path.home() / "logs" / "job-miss-check.log"
HEARTBEAT_PATH = Path("/tmp/job-miss-check-heartbeat")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "8558481275")


def latest_mtime(pattern: str) -> float | None:
    if "*" in pattern or "?" in pattern:
        matches = glob.glob(pattern)
        if not matches:
            return None
        return max(os.path.getmtime(m) for m in matches)
    if not os.path.exists(pattern):
        return None
    return os.path.getmtime(pattern)


def classify(age_hours: float | None, max_age_hours: float, period_hours: float) -> str:
    if age_hours is None:
        return "MISSED"
    if age_hours <= max_age_hours:
        return "OK"
    if age_hours <= max_age_hours + period_hours:
        return "LATE"
    return "MISSED"


def check_jobs(jobs: list[dict], now: datetime) -> list[dict]:
    results = []
    for job in jobs:
        if not job.get("enabled", True):
            continue
        mtime = latest_mtime(job["artifact"])
        age_hours = (now.timestamp() - mtime) / 3600 if mtime is not None else None
        status = classify(age_hours, job["max_age_hours"], job["period_hours"])
        results.append({
            "name": job["name"],
            "status": status,
            "age_hours": age_hours,
            "last_seen": (
                datetime.fromtimestamp(mtime, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
                if mtime is not None else "never"
            ),
        })
    return results


def send_telegram_alert(results: list[dict]) -> bool:
    if not TELEGRAM_BOT_TOKEN:
        return False
    problems = [r for r in results if r["status"] != "OK"]
    lines = [f"\U0001f6a8 Scheduled job miss detection — {len(problems)} job(s) not OK:"]
    for r in problems:
        age = f"{r['age_hours']:.1f}h" if r["age_hours"] is not None else "n/a"
        lines.append(f"  {r['status']}: {r['name']} (last seen: {r['last_seen']}, age {age})")
    message = "\n".join(lines)
    payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message}).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"Telegram send failed: {exc}", file=sys.stderr)
        return False


def main() -> None:
    now = datetime.now().astimezone()
    results = check_jobs(JOBS, now)
    problems = [r for r in results if r["status"] != "OK"]

    alert_sent = False
    if problems:
        alert_sent = send_telegram_alert(results)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ok = len(results) - len(problems)
    late = sum(1 for r in problems if r["status"] == "LATE")
    missed = sum(1 for r in problems if r["status"] == "MISSED")
    status_line = (
        f"{now.strftime('%Y-%m-%d %H:%M:%S')} [job-miss-check] "
        f"checked={len(results)} ok={ok} late={late} missed={missed} "
        f"alert_sent={alert_sent}"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(status_line + "\n")
        for r in problems:
            f.write(f"    {r['status']}: {r['name']} (last seen: {r['last_seen']})\n")
    print(status_line)
    for r in results:
        print(f"  {r['status']}: {r['name']} (last seen: {r['last_seen']})")

    HEARTBEAT_PATH.write_text(now.isoformat() + "\n")


if __name__ == "__main__":
    main()
