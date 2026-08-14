"""Tests for job_miss_check.py, including the bo-loop-heartbeat-and-slos
canary entry added to job_miss_check_config.JOBS.

The build spec's own completion criteria requires a named test proving "a
canary that never starts still alerts" — the whole point of this checker is
that it is a read-only mtime observer external to the canary process itself,
so absence (never ran) and death-mid-pipeline (ran, didn't finish) must both
read as MISSED/LATE, never as OK-by-default.
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
import job_miss_check as jmc
import job_miss_check_config as jmc_config


def _canary_job_entry():
    return next(j for j in jmc_config.JOBS if j["name"] == "bo-loop-heartbeat-canary")


def test_canary_job_entry_present_in_config():
    entry = _canary_job_entry()
    assert entry["enabled"] is True
    assert entry["artifact"].endswith("terminal-stage-heartbeat.json")


def test_canary_that_never_ran_classifies_missed(tmp_path):
    artifact = tmp_path / "terminal-stage-heartbeat.json"  # never created
    job = {**_canary_job_entry(), "artifact": str(artifact)}
    now = datetime.now().astimezone()
    results = jmc.check_jobs([job], now)
    assert results[0]["status"] == "MISSED"
    assert results[0]["last_seen"] == "never"


def test_canary_that_died_mid_pipeline_days_ago_classifies_missed(tmp_path):
    """Simulates a canary that ran once, then stopped completing (died
    mid-pipeline every day since) — same observable signature as never
    having run at all beyond the grace window, by design (see module docstring)."""
    artifact = tmp_path / "terminal-stage-heartbeat.json"
    artifact.write_text("{}")
    old_time = (datetime.now() - timedelta(days=5)).timestamp()
    os.utime(artifact, (old_time, old_time))
    job = {**_canary_job_entry(), "artifact": str(artifact)}
    now = datetime.now().astimezone()
    results = jmc.check_jobs([job], now)
    assert results[0]["status"] == "MISSED"


def test_canary_fresh_within_the_day_classifies_ok(tmp_path):
    artifact = tmp_path / "terminal-stage-heartbeat.json"
    artifact.write_text("{}")
    job = {**_canary_job_entry(), "artifact": str(artifact)}
    now = datetime.now().astimezone()
    results = jmc.check_jobs([job], now)
    assert results[0]["status"] == "OK"


def test_canary_slightly_overdue_classifies_late_not_missed(tmp_path):
    artifact = tmp_path / "terminal-stage-heartbeat.json"
    artifact.write_text("{}")
    entry = _canary_job_entry()
    overdue_hours = entry["max_age_hours"] + 1
    old_time = (datetime.now() - timedelta(hours=overdue_hours)).timestamp()
    os.utime(artifact, (old_time, old_time))
    job = {**entry, "artifact": str(artifact)}
    now = datetime.now().astimezone()
    results = jmc.check_jobs([job], now)
    assert results[0]["status"] == "LATE"
