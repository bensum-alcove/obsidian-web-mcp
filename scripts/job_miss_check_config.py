"""job_miss_check_config.py — Declarative job list for job_miss_check.py.

Add a job by appending a dict here — no changes to the checker script needed.

Fields:
  name           Human-readable identifier, used in log lines and alerts.
  artifact       Path to the file the job writes on success. May contain a
                 glob ("*") for date-stamped filenames — the newest match by
                 mtime is used. No match (or a literal path that doesn't
                 exist) is treated as "never ran".
  max_age_hours  OK if the artifact is no older than this. Set to the job's
                 normal expected age at check time (period + a grace margin),
                 not the raw period — the checker runs once daily at a fixed
                 time, not right after each job.
  period_hours   The job's own schedule period. Used only to size the LATE
                 vs. MISSED boundary: LATE up to (max_age_hours + period_hours),
                 MISSED beyond that (a full extra period overdue).
  enabled        Set False to keep a job in the inventory without alerting.
"""

JOBS = [
    {
        # Sundays 06:00 AEST. Checked daily, so a healthy run is up to ~7d old.
        "name": "contradiction-lint",
        "artifact": "/home/ben_sum/vaults/bs-brain/BS 2nd Brain/Alcove/Infrastructure/contradiction-lint-report.md",
        "max_age_hours": 174,  # 7d + 6h grace
        "period_hours": 168,
        "enabled": True,
    },
    {
        # Daily 15:30 AEST, BS Brain. Checked at 10:00 next day (~18.5h old when healthy).
        "name": "dreaming-bs-brain",
        "artifact": "/home/ben_sum/vaults/bs-brain/BS 2nd Brain/Alcove/Infrastructure/dreaming-reports/*.md",
        "max_age_hours": 20,
        "period_hours": 24,
        "enabled": True,
    },
    {
        # Daily 15:30 AEST, CB Brain. dreaming.py's report_path_for() only uses
        # the BS-Brain-style Infrastructure path for vault_name == "bs-brain";
        # every other vault gets _Reports/dreaming/ at the vault root.
        "name": "dreaming-cb-brain",
        "artifact": "/home/ben_sum/vaults/cb-brain/_Reports/dreaming/*.md",
        "max_age_hours": 20,
        "period_hours": 24,
        "enabled": True,
    },
    {
        # Daily 15:30 AEST, Alcove Brain. Same _Reports/dreaming/ convention as cb-brain.
        "name": "dreaming-alcove-brain",
        "artifact": "/home/ben_sum/vaults/alcove-brain/_Reports/dreaming/*.md",
        "max_age_hours": 20,
        "period_hours": 24,
        "enabled": True,
    },
    {
        # Daily 15:00 AEST. Checked at 10:00 next day (~19h old when healthy).
        "name": "vault-manifest",
        "artifact": "/home/ben_sum/vaults/bs-brain/_manifest.json",
        "max_age_hours": 21,
        "period_hours": 24,
        "enabled": True,
    },
    {
        # Weekly, Monday 02:00 AEST.
        "name": "retrieval-eval",
        "artifact": "/home/ben_sum/vaults/bs-brain/BS 2nd Brain/Alcove/Infrastructure/retrieval-eval/*-report.md",
        "max_age_hours": 174,  # 7d + 6h grace
        "period_hours": 168,
        "enabled": True,
    },
    {
        # Daily 23:00 AEST. Checked at 10:00 next day (~11h old when healthy).
        # Log is appended on every fire (success or failure), so freshness here
        # means "the cron fired at all", independent of pipeline outcome.
        "name": "alcovestats-export",
        "artifact": "/home/ben_sum/logs/alcovestats-cron.log",
        "max_age_hours": 13,
        "period_hours": 24,
        "enabled": True,
    },
    {
        # Every 5 minutes — the checker's own bootstrap dependency: mem-monitor
        # is the canary for "cron isn't running at all" on this box.
        "name": "mem-monitor-heartbeat",
        "artifact": "/home/ben_sum/logs/mem-monitor.log",
        "max_age_hours": 1,
        "period_hours": 1,
        "enabled": True,
    },
    {
        # Daily 08:00 AEST. Checked at 10:00 same day (~2h old when healthy).
        "name": "daily-triage",
        "artifact": "/home/ben_sum/logs/daily-triage.log",
        "max_age_hours": 4,
        "period_hours": 24,
        "enabled": True,
    },
    {
        # Daily 07:30 AEST, BS Brain. Checked at 10:00 same day (~2.5h old when healthy).
        "name": "hot-md-curate",
        "artifact": "/home/ben_sum/vaults/bs-brain/BS 2nd Brain/Alcove/Infrastructure/hot-md-reports/*.md",
        "max_age_hours": 25,
        "period_hours": 24,
        "enabled": True,
    },
    {
        # bo-loop-heartbeat-and-slos: daily synthetic canary, 20:40 AEST every
        # day (not just weekdays — the canary is synthetic and needs no real
        # trading session). Checked at 10:00 next day (~13h old when healthy).
        # Written ONLY when canary.py's run_daily_canary() completes all seven
        # stages (audit emit -> parse -> proposal -> shadow-adjudicate ->
        # shadow-remediate -> completion-review input) — a canary that dies
        # mid-pipeline leaves this exactly as stale as one that never ran at
        # all, and this checker cannot and does not need to tell the two apart.
        # This is the terminal-stage assertion required by the build spec:
        # the canary process reporting its own success is never the sole
        # authority that the loop was observed alive — this externally-owned,
        # read-only mtime check is.
        "name": "bo-loop-heartbeat-canary",
        "artifact": "/home/ben_sum/.build-orchestrator/canary/terminal-stage-heartbeat.json",
        "max_age_hours": 14,
        "period_hours": 24,
        "enabled": True,
    },
]
