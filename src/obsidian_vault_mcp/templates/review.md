---
version: '1'
kind: review
---
# {{ review_id }} — Review: {{ title }}

## What this means
Independently review the exact implementation candidate for {{ feature_id }} before deployment. Record the provider-reported identity. Effort is a request, never a gate.

## Start gate
Fetch {{ candidate_ref }} and record its exact SHA as reviewed_sha. Review only that candidate in an isolated scratch checkout. Never alter implementation code or the candidate ref. Apply the live registry's independent-review rules.

## Probes
{{ probes }}

Reproduce the witnesses and attack cases independently. Inspect the producer's evidence and executor results. Do not equate a closed producer or review row with approval.

## Scope
{{ scope }}

## Proof
{{ proof }}

Test the candidate's guards and recorded mutations. Report findings by severity with reproducible evidence. BLOCKER or HIGH findings require CHANGES_REQUIRED and status fail.

## Out of scope
{{ out_of_scope }}

## Parity
List the explicit focused test files run on both base and candidate, their executor ids, and zero-new-failure comparison. Do not repeat broad parity without missing evidence or a named new regression path.

## Output
Write the full review artifact to {{ review_artifact }} and the local-first pending log for {{ review_id }}. End both with exactly one fenced review-result YAML block. Its literal build_id must be {{ review_id }}. Include reviewed_build_id: {{ build_id }}, reviewer_build_id: {{ review_id }}, reviewed_sha, verdict, status, blocker_count, high_count, model_verified, and the actual reviewer model. APPROVED requires status pass, zero blockers/highs and verified identity. Follow whole-file heartbeat/wrap-up and house output style.

{{ summary_instruction }}

## Checkout write policy
Read-only review: do not create, modify, delete, rename or move any file in any checkout, and do not switch, detach or reset any HEAD. Write reports only to the declared pending log, review artifact, and summary; use scratch outside the checkout. Preserve initial dirt and never use git clean.
