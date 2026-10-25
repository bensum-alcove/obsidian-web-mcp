---
version: '1'
kind: deploy
source: vault-mcp-lock-pin-deploy-v1.md
---
# {{ deploy_id }} — Deploy: {{ title }}

## What this means
Ship the exact approved {{ feature_id }} candidate to the vault servers and dreaming checkout. This formula is derived from vault-mcp-lock-pin-deploy-v1. Report the actual provider identity. Effort is a request, never a gate.

## Start gate
Do not overlap a BO deploy: re-check every five minutes for up to three hours, then fail before switching if it still runs. Fetch origin. Both the {{ review_id }} local-first output and {{ review_artifact }} must contain the same APPROVED/pass review-result, zero blockers/highs and model_verified true. reviewed_sha must equal D = {{ candidate_ref }}. Closure alone is never approval. Read canonical checkout records and SYSTEM-FACTS, confirm each program's live directory/command, live release R and dreaming HEAD. R must be an ancestor of D. Record initial dirty paths, program PIDs/uptimes and health on 8420, 8423, 8426 and cb-brain-marketing-ro.

## Probes
{{ probes }}

Dry-run all post-switch comparisons against R read-only or a credential-free scratch copy. Never retrieve a runtime credential or copy credential-bearing configuration.

## Scope
{{ scope }}

## Procedure
1. Record rollback R and back up non-secret release/checkout pointers, package inventories and cron definitions in this build's backup area. Preserve pre-existing dirt in both checkouts.
2. Before switching, run uv sync --frozen --dry-run from a scratch clone of D against a scratch copy of the live venv. Only the exact package differences approved by review may proceed. Never run a non-dry-run dependency sync against the live venv. For a lock-only change the package inventory must be byte-identical.
3. Run the producer/reviewer focused tests on D in a clean scratch install through the structured executor, explicit files. Verify the exact review SHA again before switching.
4. Non-force fast-forward the live checkout to D. With supervisorctl, reread/update/restart alcove-brain-vault, then cb-brain-vault, then bs-brain-vault, one at a time. Each health endpoint must return 200 and its vault_read plus vault_search canaries must pass before moving to the next. Do not restart cb-brain-marketing-ro; verify it stays RUNNING and healthy. Confirm the live package inventory matches the approved expectation.
5. Non-force fast-forward the dreaming checkout to D while preserving its dirt. If its cron uses uv run, run one approved read-only cron script exactly as cron does and record its package inventory and any reviewed changes.
6. Any health/canary/parity failure restores the live release pointer to R using the recorded rollback procedure and the same one-at-a-time supervised restart order, verifies all health checks, then stops with fail. Never kill processes by pattern.
7. Insert the infrastructure changelog entry with a targeted replace anchored on the newest heading; update canonical checkout records with targeted edits. No unrelated vault rewrite.

## Proof
{{ proof }}

Record exact review/candidate/live/dreaming SHAs, rollback, package dry-run and before/after inventory, executor ids, service PIDs, health and canaries. Only small pre-existing test-fixture or non-logic wiring fixes of at most about 30 lines may be separately committed FIX- in scratch and rechecked. Never change server behavior, dependencies, assertions, thresholds or deadlines during deploy. Larger remediation stops before switching for a lawful successor.

## Out of scope
{{ out_of_scope }}

## Output
Write this build's pending log and machine summary with explicit pass/fail and one plain-English line. Follow house style and whole-file heartbeat/wrap-up rules. Attempt final attestation; session-not-quiescent means deferred_to_bo. Exit after reporting.

{{ summary_instruction }}

## Checkout write policy
No source edits in any checkout except separately permitted FIX- commits in scratch. Only scratch clones, reviewed non-force checkout fast-forwards, targeted release pointers and the named supervised restarts are allowed. Never sync the live venv, use git clean, kill by pattern, or touch pre-existing dirt.
