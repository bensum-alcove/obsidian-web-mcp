"""SLO/SLI registry for BS Brain (and sibling vaults') operational health
(vault-observability-slo build).

This module defines WHAT is measured and WHAT counts as warning/critical for
each signal -- it does not collect the values itself. Collectors are
scripts/vault_functional_canary.py (functional read/query/write/patch/
cleanup) and scripts/vault_observability_status.py (the aggregator, which
also pulls in job_miss_check's existing per-job freshness data, dreaming's
report state, contradiction_lint's count, canonical_state_scan's duplicate/
malformed counts, and vault-backup.sh's per-vault backup-age state).

Direction semantics:
  "above"  -- warning/critical are lower bounds; breach when value >= threshold
              (e.g. an age in hours: bigger is worse).
  "below"  -- warning/critical are upper bounds; breach when value <= threshold
              (e.g. a retrieval score: smaller is worse).
  "equals" -- any value other than the expected "ok" sentinel is a breach at
              the critical level (e.g. a count that must be exactly zero).

Thresholds below are backfilled from the concrete evidence cited in each
SLI's `baseline_evidence` field -- see vault-observability-slo-output.md for
the full backfill rationale. Where no real incident/measurement history
exists yet (a brand new signal), the threshold is set conservatively and
flagged `baseline_evidence="none -- new signal, revisit after real data"` so
it is never mistaken for an evidence-backed number.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    ABOVE = "above"   # bigger value = worse
    BELOW = "below"   # smaller value = worse
    EQUALS = "equals"  # anything other than the ok_value is critical


class Status(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"  # no data collected this run -- never silently treated as OK


@dataclass(frozen=True)
class SLI:
    id: str
    description: str
    layer: str  # groups related signals for "layer-specific failure reporting"
    unit: str
    direction: Direction
    warning: float | None
    critical: float | None
    ok_value: float | str | None = None  # only meaningful for Direction.EQUALS
    owner: str = "ben"
    runbook: str = ""
    baseline_evidence: str = ""

    def evaluate(self, value: float | str | None) -> Status:
        if value is None:
            return Status.UNKNOWN
        if self.direction is Direction.EQUALS:
            return Status.OK if value == self.ok_value else Status.CRITICAL
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return Status.UNKNOWN
        if self.direction is Direction.ABOVE:
            if self.critical is not None and numeric >= self.critical:
                return Status.CRITICAL
            if self.warning is not None and numeric >= self.warning:
                return Status.WARNING
            return Status.OK
        # Direction.BELOW
        if self.critical is not None and numeric <= self.critical:
            return Status.CRITICAL
        if self.warning is not None and numeric <= self.warning:
            return Status.WARNING
        return Status.OK


REGISTRY: dict[str, SLI] = {}


def _register(sli: SLI) -> SLI:
    REGISTRY[sli.id] = sli
    return sli


mcp_availability = _register(SLI(
    id="mcp_availability",
    description="Vault MCP process/HTTP liveness (localhost + cloudflared + external URL).",
    layer="process",
    unit="watchdog_restarts_per_check",
    direction=Direction.EQUALS,
    warning=None,
    critical=None,
    ok_value=0,
    runbook="check-vault-mcp.sh already self-heals (restarts supervisord program); "
            "this SLI just makes repeated restarts visible instead of silently absorbed.",
    baseline_evidence="check-vault-mcp.sh has run every 2 minutes per vault since "
                       "before this build with no counted-restart history kept -- "
                       "0 restarts/check is the only defensible starting baseline.",
))

functional_read_query = _register(SLI(
    id="functional_read_query",
    description="vault_functional_canary.py: read/query/write/patch/cleanup round trip, per layer.",
    layer="functional",
    unit="layers_failing",
    direction=Direction.ABOVE,
    warning=1,
    critical=2,
    runbook="Inspect the failing layer name in the canary's per-run JSON "
            "(~/.local/state/vault-observability/canary-<vault>.json) -- each "
            "layer fails independently, never collapsed to red/green.",
    baseline_evidence="none -- new signal, revisit after real data. Set 1 layer "
                       "failing = warning (could be a transient lock timeout under "
                       "vault_lock's 30s default) and 2+ = critical (a systemic break) "
                       "as a conservative starting point.",
))

index_freshness_seconds = _register(SLI(
    id="index_freshness_seconds",
    description="Age of the frontmatter-index parse confirmation in the last canary run.",
    layer="functional",
    unit="seconds",
    direction=Direction.ABOVE,
    warning=1800,
    critical=7200,
    runbook="Confirm the canary cron is still firing (job_miss_check covers this) "
            "before assuming the index itself is broken.",
    baseline_evidence="none -- new signal. Canary is intended to run every 15 "
                       "minutes (matches vault-backup.sh's cadence); 2x and 8x "
                       "that period are the warning/critical starting points.",
))

backup_age_hours = _register(SLI(
    id="backup_age_hours",
    description="Hours since vault-backup.sh last recorded a real change+push for this vault.",
    layer="durability",
    unit="hours",
    direction=Direction.ABOVE,
    warning=24,
    critical=48,
    runbook="vault-backup.sh already alerts at 48h via its own STALE_THRESHOLD_SECONDS; "
            "this SLI surfaces the same signal on the dashboard, it does not replace "
            "vault-backup.sh's own alerting.",
    baseline_evidence="vault-backup.sh's existing STALE_THRESHOLD_SECONDS=172800 (48h) "
                       "-- critical here matches it exactly; warning set at half that "
                       "so the dashboard goes amber before vault-backup.sh's own "
                       "Telegram alert would fire.",
))

restore_drill_age_days = _register(SLI(
    id="restore_drill_age_days",
    description="Days since the last documented clean-room restore-drill proof.",
    layer="durability",
    unit="days",
    direction=Direction.ABOVE,
    warning=60,
    critical=120,
    runbook="Run a clean-room restore drill (see ~/backups/vault-clean-room-restore-proof-*/ "
            "for the proof-artifact shape from the last one) and archive its proof directory.",
    baseline_evidence="Last real restore drill: ~/backups/vault-clean-room-restore-proof-20260817 "
                       "(2026-08-17). No established recurring cadence exists yet -- 60/120 days "
                       "is a conservative starting cadence, not evidence of a target SLA.",
))

retrieval_trend = _register(SLI(
    id="retrieval_trend",
    description="Week-over-week delta in vault-retrieval-eval's R@5/MRR score.",
    layer="quality",
    unit="score_delta",
    direction=Direction.BELOW,
    warning=-0.05,
    critical=-0.15,
    runbook="Inspect BS 2nd Brain/Alcove/Infrastructure/retrieval-eval/*-report.md "
            "for the two most recent weekly runs.",
    baseline_evidence="vault-retrieval-eval-v3 runs weekly (Mondays 02:00 UTC) -- no "
                       "regression has yet been observed to calibrate a tighter number "
                       "against; -0.05/-0.15 are conservative starting deltas.",
))

validation_rejects = _register(SLI(
    id="validation_rejects_count",
    description="Malformed canonical-state records found by canonical_state_scan.py.",
    layer="integrity",
    unit="count",
    direction=Direction.ABOVE,
    warning=1,
    critical=5,
    runbook="Run scripts/canonical_state_scan.py and fix the malformed record(s) it lists.",
    baseline_evidence="canonical_state_scan.py is report-only and exit-code-gated at "
                       "\"any malformed record\" already -- warning=1 matches that "
                       "existing bar exactly.",
))

concurrency_conflicts = _register(SLI(
    id="concurrency_conflicts_count",
    description="vault_lock conflict-skips recorded in the mutation ledger since the last check.",
    layer="integrity",
    unit="count_per_day",
    direction=Direction.ABOVE,
    warning=3,
    critical=10,
    runbook="Read ~/.build-orchestrator/ledgers/dreaming-mutations.jsonl for "
            "status==conflict_skipped entries and identify the concurrent writer.",
    baseline_evidence="dreaming-safe-remediation-v2's shadow run across all three "
                       "live vaults found 0 concurrency conflicts -- any sustained "
                       "rate above single digits/day would be a new, unexplained pattern.",
))

malformed_notes = _register(SLI(
    id="malformed_notes_count",
    description="Notes with unparsable YAML frontmatter, from dreaming.py's nightly report.",
    layer="integrity",
    unit="count",
    direction=Direction.ABOVE,
    warning=1,
    critical=10,
    runbook="Read the latest dreaming report (BS 2nd Brain/Alcove/Infrastructure/"
            "dreaming-reports/*.md or vault-root _Reports/dreaming/*.md) for the "
            "malformed-frontmatter finding list.",
    baseline_evidence="none -- new signal, revisit once dreaming.py's report format "
                       "is parsed by the aggregator for real counts.",
))

contradiction_count = _register(SLI(
    id="contradiction_count",
    description="SYSTEM-FACTS contradictions found by the weekly contradiction_lint.py pass.",
    layer="integrity",
    unit="count",
    direction=Direction.ABOVE,
    warning=1,
    critical=5,
    runbook="Read BS 2nd Brain/Alcove/Infrastructure/contradiction-lint-report.md "
            "and resolve the listed contradiction(s) -- this class is FORBIDDEN from "
            "automation (dreaming-safe-remediation-v2), always a human judgment call.",
    baseline_evidence="contradiction_lint.py is already report-only with no prior "
                       "count history summarized centrally -- warning=1 treats any "
                       "contradiction as worth surfacing, matching its own report-or-nothing design.",
))

dreaming_state = _register(SLI(
    id="dreaming_state",
    description="Whether last night's dreaming.py report-only cycle completed for this vault.",
    layer="scheduled_jobs",
    unit="bool",
    direction=Direction.EQUALS,
    warning=None,
    critical=None,
    ok_value="completed",
    runbook="Already monitored by job_miss_check.py's dreaming-<vault> job entries; "
            "this SLI re-exposes the same signal for the dashboard, see job_miss_check_config.py.",
    baseline_evidence="job_miss_check_config.py's dreaming-* jobs already define "
                       "max_age_hours=20/period_hours=24 for this exact artifact.",
))

hot_md_policy = _register(SLI(
    id="hot_md_policy_state",
    description="Whether hot.md is within its curated character budget (hot-md-curate.py).",
    layer="scheduled_jobs",
    unit="bool",
    direction=Direction.EQUALS,
    warning=None,
    critical=None,
    ok_value="within_budget",
    runbook="Run scripts/hot-md-curate.py --apply (already scheduled daily 07:30 AEST) "
            "or inspect its report for what pushed hot.md over HOT_MD_BUDGET_CHARS.",
    baseline_evidence="config.HOT_MD_BUDGET_CHARS=5000 is the existing single source "
                       "of truth shared by hot-md-curate.py and dreaming.py's own "
                       "hot_md_budget finding type.",
))


def layers() -> list[str]:
    """Distinct layer names across the registry, in registration order."""
    seen: list[str] = []
    for sli in REGISTRY.values():
        if sli.layer not in seen:
            seen.append(sli.layer)
    return seen
