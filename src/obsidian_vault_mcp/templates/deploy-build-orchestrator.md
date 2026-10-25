---
version: '2'
kind: deploy
source: bo-escalation-allowlist-deploy-v2.md
---
# {{ deploy_id }} — Deploy: {{ title }}

## What this means
Ship the exact approved candidate for {{ feature_id }} through the BO release procedure. This formula is derived from bo-escalation-allowlist-deploy-v2 and its inherited deploy gates. Report the actual provider identity. Effort is a request, never a gate.

## Start gate
Fetch origin. Resolve the ACHIEVED review goal {{ review_id }} for project {{ project }}, root {{ review_id }}, through BO's read-only goal state (or a credential-free extraction). Let A = achieved_by_build_id. Require A to be a closed reviewer in that exact goal lineage. Read A's own frozen spec to resolve the artifact it writes (the original review normally writes {{ review_artifact }}; a generated re-review writes its own artifact). Never substitute the original failed review's artifact. The final review-result blocks in A's own local-first output and its own artifact must agree: build_id and reviewer_build_id equal A, APPROVED, status pass, blocker_count 0, high_count 0, model_verified true, and a reviewer accepted by the live independent-review rules. Their reviewed_sha must equal D = {{ candidate_ref }} after fetching. Missing, ambiguous, stale or mismatched evidence stops before switching. Closure alone is never approval. Record live release R, main M, required program PIDs/uptimes and public/port health baselines. Never infer approval from review closure.

## Probes
{{ probes }}

Dry-run every post-switch probe before switching against R read-only or a credential-free extraction. Never copy the live database or retrieve credentials. A verifier setup error may be corrected up to three times per check, 45 minutes total, without weakening the comparison.

## Scope
{{ scope }}

## Procedure
1. Follow the BO release procedure and deploy kit inherited by Personal/Build Orchestrator/specs/bo-escalation-allowlist-deploy-v2.md, substituting this build's ids, candidate and approved review. Back up non-secret release pointers and the governed cron definitions in this build's backup area; record rollback R. Never copy secrets or credential-bearing files.
2. Apply the kit freshness gate. Only a clean reviewed integration proceeds to all gates; integration-required or unexpected-live-release stops before switching. Never force-push.
3. Gate A: run the approved producer/reviewer focused parity files through the structured executor in explicit-file batches. Gate B: replay every producer/fix mutation in scratch; each must turn a test red. Any required broader integration suite needs its governed scope grant.
4. For changed DB tables, use tools/bo_db_extract.py's credential-free extraction and integrity_check; prove normal startup migrates once and reads back. Live db.init_db runs only through normal startup.
5. Switch orchestrator, orchestrator-dashboard and bs-control-session-bridge using immutable releases and the governed reread, update, restart, status procedure. Switch log-syncer only if its code or imported modules changed. Keep bs-control-mcp unswitched unless bs_control_mcp/ changed; then perform its governed Phase B and MCP canaries. Record both decisions.
6. Verify release parity on D, dashboard/login health, bo_status, public baselines and every row running before the switch remaining running or normally finished on its frozen model. Watch health for 20 minutes. Use read-only or stubbed-delivery canaries, never a real alarm.
{{ deploy_canaries }}7. A correctness, health or parity failure rolls the switched programs and governed cron definitions back to R, verifies health, and stops with fail. Reviewer-accepted non-blocking findings proceed and are documented. Any remediation beyond small non-logic fixture/wiring fixes requires the lawful successor; never change logic or weaken tests during deploy.
8. After every gate passes, non-force fast-forward main to the exact final approved D. Update the infrastructure changelog with a targeted insertion and compiled current-state records with targeted edits. Preserve unrelated vault bytes.

## Proof
{{ proof }}

Record R, M, D, the approval source, integration record, rollback ids, executor ids, mutations, schema proof, service decisions and every canary. Never count a verifier that failed before comparison as a passing check.

## Out of scope
{{ out_of_scope }}

## Output
Write this build's pending log and summary with an explicit pass/fail. Follow the house style and whole-file heartbeat/wrap-up rules. Attempt final worktree attestation; session-not-quiescent means deferred_to_bo. Exit after reporting.

{{ summary_instruction }}

## Checkout write policy
Do not edit source or tests except separately permitted small FIX- commits in scratch. Only the reviewed release procedure, immutable new release directories and final non-force main fast-forward are allowed. Never edit an existing release directory, use git clean, or touch pre-existing dirt.
