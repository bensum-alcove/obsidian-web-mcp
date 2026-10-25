---
version: '2'
kind: build
---
# {{ build_id }} — {{ title }}

## What this means
{{ what_this_means }}

This is the implementation stage of {{ feature_id }}. Deployment is deferred until the exact candidate receives independent approval. This stage owns the implementation goal; a generated repair retains the candidate ref and the review goal tracks its achieving approval. Report the provider identity that actually ran. Effort is a request, never a gate.

## Start gate
Use the BO isolated worktree and the configured project base. Fetch origin and record the base SHA and initial dirty paths. Preserve pre-existing dirt. Never write the live or dreaming checkout.

## Probes
{{ probes }}

Record witnesses, negative controls and attack cases before changing code. Stop with a failing report if a required start gate cannot be established.

## Scope
{{ scope }}

## Proof
{{ proof }}

Run the focused changed tests and direct caller tests on the base and candidate through BO's structured test executor, using explicit files and disk scratch. Zero new failing nodes. Broad parity: no unless separately governed. Exercise each safety guard with a mutation in scratch.

## Out of scope
{{ out_of_scope }}

## Commit and publish
Prefix commits {{ commit_prefix }} and push non-force to {{ candidate_ref }} after each group. Do not integrate or deploy this stage.

## Output
Write the local-first pending log for this build, including base/final SHAs, probes, changes at file:line, test executor ids, parity and mutations. Write the required summary. Follow the house output style and whole-file heartbeat/wrap-up rules in Personal/Build Orchestrator/instructions/01-spec-authoring-guide.md.

{{ summary_instruction }}

## Checkout write policy
Source and test edits within this implementation's scope are allowed and required. Commit in the isolated worktree and push only to {{ candidate_ref }}. Keep reports in the pending-log area and scratch outside the checkout. Preserve initial dirt, leave no new dirt, and never use git clean.
