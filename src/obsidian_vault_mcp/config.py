import os
from pathlib import Path

# Vault configuration
VAULT_PATH = Path(os.environ.get("VAULT_PATH", os.path.expanduser("~/Obsidian/MyVault")))
VAULT_MCP_TOKEN = os.environ.get("VAULT_MCP_TOKEN", "")
TEAMBOT_MCP_TOKEN = os.environ.get("TEAMBOT_MCP_TOKEN", "")
VAULT_MCP_PORT = int(os.environ.get("VAULT_MCP_PORT", "8420"))

# OAuth 2.1 password gate (opt-in — leave unset for no auth gate)
VAULT_AUTH_PASSWORD = os.environ.get("VAULT_AUTH_PASSWORD", "")
VAULT_BASE_URL = os.environ.get("VAULT_BASE_URL", "")  # e.g. https://vault.bensum.org

# OAuth 2.1 client credentials (required when VAULT_AUTH_PASSWORD is set)
VAULT_OAUTH_CLIENT_ID = os.environ.get("VAULT_OAUTH_CLIENT_ID", "")
VAULT_OAUTH_CLIENT_SECRET = os.environ.get("VAULT_OAUTH_CLIENT_SECRET", "")

# OAuth token persistence (opt-in — leave unset for in-memory tokens, byte-identical
# to pre-persistence behaviour: 24h access TTL, no refresh_token grant)
VAULT_TOKEN_DB = os.environ.get("VAULT_TOKEN_DB", "")
VAULT_TOKEN_TTL_SECONDS = int(
    os.environ.get("VAULT_TOKEN_TTL_SECONDS", "2592000" if VAULT_TOKEN_DB else "86400")
)  # 30 days when persisted, 24h in-memory default (unchanged)
VAULT_REFRESH_TTL_SECONDS = int(os.environ.get("VAULT_REFRESH_TTL_SECONDS", "7776000"))  # 90 days

# Safety limits
MAX_CONTENT_SIZE = 1_000_000  # 1MB max write size
MAX_BATCH_SIZE = 20           # Max files per batch operation
MAX_SEARCH_RESULTS = 50       # Max results per search
DEFAULT_SEARCH_RESULTS = 20
MAX_LIST_DEPTH = 5            # Max directory recursion depth
CONTEXT_LINES = 2             # Default lines of context in search results

# Directories to never expose or modify
EXCLUDED_DIRS = {".obsidian", ".trash", ".git", ".DS_Store", ".semantic-index", ".mutation-ledger"}

# Dedicated scratch namespace for synthetic monitoring (see
# scripts/vault_functional_canary.py, vault-observability-slo build).
# Deliberately NOT added to EXCLUDED_DIRS: the frontmatter-index parse path
# and vault_list must still be able to see it -- the functional canary's
# "verify index sees it" step needs the real frontmatter-parsing code path
# to observe scratch writes, and an operator needs vault_list to inspect
# canary state for debugging. Instead it is excluded only from *ordinary
# retrieval* (full-text search, semantic search) via RETRIEVAL_EXCLUDED_DIRS
# below, so synthetic canary writes never surface in real search results but
# are not invisible to the tooling that needs to see them.
SCRATCH_DIR_NAME = "_scratch"

# vault-retrieval-r5-085-v1: retrieval-eval report directories quote every
# benchmark question verbatim as rubric/top_result text (see run_eval_v3.py's
# "Manual rubric review needed" section) and accumulate one dated report per
# run. Left in ordinary retrieval, these become the single strongest keyword
# match for a large fraction of the very questions they document -- diagnosed
# as the direct cause of 16/30 frozen-v3 misses in this build's miss trace
# (candidate slots consumed by report noise, or the report itself winning the
# match outright). This isn't benchmark-specific: any real user asking one of
# these documented questions in production hits the exact same contamination.
# Same treatment as _scratch above -- hidden from ordinary retrieval only,
# still vault_list/frontmatter-index visible. No-op in vaults without these
# directories (Alcove/CB Brain at the time of this build).
EVAL_REPORT_DIR_NAMES = {"retrieval-eval", "retrieval-eval-v3"}
RETRIEVAL_EXCLUDED_DIRS = EXCLUDED_DIRS | {SCRATCH_DIR_NAME} | EVAL_REPORT_DIR_NAMES

# Frontmatter index refresh interval (seconds)
FRONTMATTER_INDEX_DEBOUNCE = 5.0

# hot.md ephemeral-cache budget in chars, frontmatter excluded. Single source of
# truth shared by scripts/hot-md-curate.py (enforcement) and scripts/dreaming.py
# (nightly report flag) -- see BS 2nd Brain/Alcove/Infrastructure/hot-md-structure.md.
HOT_MD_BUDGET_CHARS = 5000

# Rate limiting (requests per minute) -- track in-memory, enforce per-token
RATE_LIMIT_READ = 100
RATE_LIMIT_WRITE = 30

# vault_query keyword-leg tokenization (opt-out — defaults enabled). When disabled,
# the keyword leg passes the raw question straight to ripgrep as it always has, byte-
# identical to pre-tokenization behaviour. This is the fastest revert path: a
# supervisord environment edit + restart, no git operation needed.
VAULT_QUERY_KEYWORD_TOKENIZE = os.environ.get("VAULT_QUERY_KEYWORD_TOKENIZE", "1") not in (
    "0", "false", "False", "",
)

# vault_search tokenized-augmentation (opt-out — defaults enabled). Separate from
# VAULT_QUERY_KEYWORD_TOKENIZE so the two deploys can be reverted independently. When
# disabled, vault_search is byte-identical to pre-tokenization behaviour. See
# tools/search.py's _augment_with_tokenized_matches for the exact-string-verification
# contract this guards.
VAULT_SEARCH_TOKENIZE = os.environ.get("VAULT_SEARCH_TOKENIZE", "1") not in (
    "0", "false", "False", "",
)

# vault-retrieval-candidate-recall-v1: opt-out kill switch for the partial-match
# keyword-leg candidate (see tools/search.py's _search_by_tokens allow_partial
# param). Default enabled. When disabled, the AND gate reverts to its previous
# all-or-nothing behaviour (a full match set, or nothing at all).
VAULT_QUERY_ALLOW_PARTIAL_KEYWORD_MATCH = os.environ.get(
    "VAULT_QUERY_ALLOW_PARTIAL_KEYWORD_MATCH", "1"
) not in ("0", "false", "False", "")

# Write-contract gate mode: "off" | "shadow" (default) | "enforce". See
# write_contract.py for the full contract. Read directly from the environment
# there (not this module) so the mode can be flipped without importing config
# in a hot loop; surfaced here purely so every runtime flag is documented in
# one place. Fastest revert path: env var edit + supervisorctl restart.
VAULT_WRITE_CONTRACT_MODE = os.environ.get("VAULT_WRITE_CONTRACT_MODE", "shadow").strip().lower()

# Optimistic-concurrency mode for revision-guarded writes: "off" | "shadow" | "enforce"
# (default enforce). See vault.py's _check_revision/_concurrency_mode for the full
# contract. Unlike the write-contract gate above, this only ever activates when a
# caller explicitly passes expected_revision to a mutation tool -- legacy callers
# that never send it are byte-identical to pre-this-feature behaviour in every mode.
# "shadow" runs the revision comparison and logs the outcome without ever blocking a
# write; "off" skips the comparison entirely (expected_revision is accepted but
# ignored). Read directly from the environment in vault.py (not this module) so the
# mode can be flipped without an import-time cache. Fastest revert path: env var edit
# + supervisorctl restart, no git operation needed.
VAULT_OPTIMISTIC_CONCURRENCY_MODE = os.environ.get("VAULT_OPTIMISTIC_CONCURRENCY_MODE", "enforce").strip().lower()

# Vault mutation ledger: "on" (default) | "off". See mutation_ledger.py for the
# full contract. A ledger failure never blocks a write in either mode -- "off"
# just skips recording entirely (fastest revert path: env var edit, no restart
# even required since the mode is read fresh on every call).
VAULT_MUTATION_LEDGER_MODE = os.environ.get("VAULT_MUTATION_LEDGER_MODE", "on").strip().lower()

# Ledger storage directory. Defaults to <VAULT_PATH>/.mutation-ledger -- a
# dot-directory, so resolve_vault_path already refuses any mutation tool from
# targeting it, and it's included in EXCLUDED_DIRS above so vault_list/
# vault_search/vault_semantic_search/vault_recent_changes/vault_stats/the
# frontmatter index all skip it like every other excluded dir.
VAULT_MUTATION_LEDGER_DIR = os.environ.get("VAULT_MUTATION_LEDGER_DIR", "")

# Ledger rotation: size-bounded (bytes) and count-bounded (number of rotated
# backups kept), via stdlib logging.handlers.RotatingFileHandler. Default caps
# total ledger disk usage at roughly 5MB * 11 files = ~55MB per vault.
VAULT_MUTATION_LEDGER_MAX_BYTES = int(os.environ.get("VAULT_MUTATION_LEDGER_MAX_BYTES", "5000000"))
VAULT_MUTATION_LEDGER_BACKUP_COUNT = int(os.environ.get("VAULT_MUTATION_LEDGER_BACKUP_COUNT", "10"))

# Build Orchestrator authoring contract adapter (vault-bo-authoring-mcp-v1). The
# bo_validate_build_graph/bo_create_build/bo_create_chain tools invoke this CLI as
# a subprocess (shell=False, JSON stdin/stdout) rather than re-encoding BO schema
# rules in this repo -- see bo_contract.py. Absent/wrong-version/failing adapter
# means those tools fail closed (no schedule activation), by design.
BO_AUTHORING_CONTRACT_PATH = os.environ.get(
    "BO_AUTHORING_CONTRACT_PATH",
    os.path.expanduser("~/build-orchestrator/authoring_contract.py"),
)
BO_AUTHORING_CONTRACT_PYTHON = os.environ.get("BO_AUTHORING_CONTRACT_PYTHON", "python3")
BO_AUTHORING_CONTRACT_TIMEOUT_SECONDS = float(os.environ.get("BO_AUTHORING_CONTRACT_TIMEOUT_SECONDS", "15"))

# Build Orchestrator path-mutation guard mode: "off" | "shadow" (default) | "enforce".
# Independent of VAULT_WRITE_CONTRACT_MODE above -- this guard is specific to
# Personal/Build Orchestrator/specs/ and Personal/Build Orchestrator/schedules/ and
# is deployed shadow-only in vault-bo-authoring-mcp-v1 per its spec (enforcement is
# a separate, later build gated on independent Codex review). See bo_guard.py.
# Fastest revert path: env var edit + supervisorctl restart, no git operation needed.
BO_PATH_GUARD_MODE = os.environ.get("BO_PATH_GUARD_MODE", "shadow").strip().lower()

# vault_query RRF fusion sharpness (kill switch: env var edit + supervisorctl restart,
# no git operation needed). vault-query-calibration-v2 diagnosed that k=60 is flat
# enough, relative to the candidate fetch depth, that a document appearing at a
# middling rank in BOTH legs routinely out-scores a document ranked #1 in only ONE
# leg (their summed 1/(k+rank) terms exceed the single leg's 1/(k+1)) -- but that
# diagnosis was never actually deployed (production still ran the default k=60; see
# vault-retrieval-r5-085-v1's miss trace, which found 7 of 17 remaining frozen-v3
# misses were this exact "confident single-leg #1 buried by consensus" pattern).
# vault-retrieval-r5-085-v1 empirically swept k against the full frozen-v3 corpus
# (not tuned per-question) and found a stable plateau at k=4..8 (34-35/45 scored
# questions hit, vs 28/45 at k=60); k=6 is the plateau's midpoint. Lowering k
# sharpens the top of the curve so a confident single-leg #1 is harder to displace,
# without changing the behaviour for genuine both-legs-agree consensus (which wins
# at any k>0 since it's always ~2x a single leg's score at the same rank).
VAULT_QUERY_RRF_K = float(os.environ.get("VAULT_QUERY_RRF_K", "6"))

# vault_query canonical-state authority boost (kill switch: 1.0 = no-op, byte-identical
# to pre-calibration behaviour). Multiplies the fused score of any result whose
# frontmatter declares `type: canonical-state` before temporal decay is applied. The
# vault's own documented precedence rule (infrastructure.md's "Current-state authority
# note") says these records are the current-state authority over general prose --
# this mechanism is a modest, multiplicative nudge toward that rule, not a hard
# override, so a genuinely stronger match can still win. Inert (no matching files) in
# any vault that has no Canonical State/records/ tree yet, e.g. CB/Alcove Brain at the
# time of this build.
#
# vault-retrieval-r5-085-v1: the previously-deployed live value (1.3, set only via
# supervisord env, never promoted to this default) was empirically almost a no-op --
# canonical-state records that were already ranked well outside a competing spec's/
# changelog's/build-log's top-5 fused rank need a multiplier proportional to that
# rank gap, not a small nudge. Swept boost against frozen-v3 (aggregate, not
# per-question) at the now-corrected k=6: 1.0/1.3 -> 0.778 R@5, plateauing at
# 3.0-5.0 -> 0.867, declining again by 15+ (confirms this is a real peak, not
# "bigger is always better" — a canonical record can still lose to a much stronger
# prose match at extreme boost, which is the intended non-absolute behaviour).
VAULT_QUERY_CANONICAL_BOOST = float(os.environ.get("VAULT_QUERY_CANONICAL_BOOST", "3.0"))

# vault_query temporal decay: half-life in days, env-overridable.
# Longest matching path substring wins; unmatched paths use the default.
VAULT_QUERY_DEFAULT_HALF_LIFE_DAYS = float(os.environ.get("VAULT_QUERY_HALF_LIFE_DAYS", "90"))
VAULT_QUERY_HALF_LIFE_OVERRIDES = {
    "Claude-Code-Prompts/": float(os.environ.get("VAULT_QUERY_HALF_LIFE_CLAUDE_CODE_PROMPTS", "30")),
    "Skills/": float(os.environ.get("VAULT_QUERY_HALF_LIFE_SKILLS", "180")),
    "Clients/": float(os.environ.get("VAULT_QUERY_HALF_LIFE_CLIENTS", "365")),
}

# vault_query entity-mention candidate leg (this build: vault-retrieval-entity-
# resolution-r5-v2). Kill switch: "0"/"false" reverts to exactly the two-leg
# (keyword + semantic) RRF fusion from vault-retrieval-r5-085-v1, no git
# operation needed. See tools/query.py's _entity_leg for the mechanism: a
# query mentioning a known entity's surname -- even buried in a longer
# descriptive sentence with no other literal overlap with the target file --
# gets that entity's own file injected as a third RRF-fused candidate leg,
# corpus-derived from _entities.json (built nightly from real vault content by
# scripts/dreaming.py, never from benchmark question text). Diagnosed against
# vault-retrieval-entity-resolution-r5-v2's holdout-2 miss trace: 5 of 9 misses
# had the correct file's own surname literally present in the query but either
# absent from the keyword leg's AND-preferred token match (buried among many
# other descriptive tokens) or ranked too deep in both legs individually to
# survive RRF fusion against documents that ranked moderately in both legs at
# once.
VAULT_QUERY_ENTITY_EXPANSION = os.environ.get("VAULT_QUERY_ENTITY_EXPANSION", "1") not in (
    "0", "false", "False", "",
)

# Cap on how many distinct entity files this leg can inject per query --
# a latency/flooding guard, not a correctness knob. Ranked by number of
# distinct query tokens matched (ties broken by _entities.json order), so if
# this cap is ever hit, the most-confidently-matched entities are kept.
VAULT_QUERY_ENTITY_EXPANSION_MAX_CANDIDATES = int(
    os.environ.get("VAULT_QUERY_ENTITY_EXPANSION_MAX_CANDIDATES", "8")
)

# vault_semantic_search chunking mechanisms (vault-retrieval-candidate-recall-v1).
# Each is independently toggleable/revertible via env var, no git operation needed.
# All default ON; disabling any one reproduces its specific piece of the previous
# (pre-this-build) chunking behaviour. A full semantic-index rebuild is required
# after flipping any of these (see scripts/rebuild_semantic_index.py) since chunk
# boundaries are a function of file content + these flags, and the index only
# re-chunks a file when its mtime/hash changes.
VAULT_SEMANTIC_STRIP_FRONTMATTER = os.environ.get("VAULT_SEMANTIC_STRIP_FRONTMATTER", "1") not in (
    "0", "false", "False", "",
)
VAULT_SEMANTIC_FRONTMATTER_PREFIX = os.environ.get("VAULT_SEMANTIC_FRONTMATTER_PREFIX", "1") not in (
    "0", "false", "False", "",
)
VAULT_SEMANTIC_TABLE_ROW_CHUNKING = os.environ.get("VAULT_SEMANTIC_TABLE_ROW_CHUNKING", "1") not in (
    "0", "false", "False", "",
)

# vault_query / vault_semantic_search candidate-pool depth before RRF fusion.
# vault-retrieval-candidate-recall-v1: raised from a bare max_results*5/10 multiplier
# to a higher floor -- diagnosis showed several correct paraphrase-category documents
# sitting at semantic rank 16-45 (see this build's output doc's rank traces), well
# past the previous ~50-75 candidate ceiling for a top_k=5-8 caller, meaning they
# were absent from fusion entirely regardless of any rerank/boost tuning. Kill switch:
# set back to the previous multiplier-derived value (unused when this constant is at
# its default derivation, see tools/query.py/tools/semantic_search.py call sites).
#
# vault-retrieval-r5-085-v1 found 100 still insufficient for some genuinely hard
# cases (e.g. an entity-alias question whose correct client file only reached
# distinct-file semantic rank 5 at candidate depth 300, invisible at depth 100) and
# for the keyword leg, where a large well-matching-filename document can consume the
# AND-match candidate slots ahead of a correct canonical-state record with weaker
# filename overlap. Raised to 200 -- empirically the full gain of raising all the
# way to the hard cap of 300 (see this build's k/depth sweep), at measured-identical
# per-query latency locally (SQLite KNN + the keyword leg's already-fixed full-vault
# scan cost dominate either way; this constant only bounds the output slice).
VAULT_SEMANTIC_FETCH_MULTIPLIER = int(os.environ.get("VAULT_SEMANTIC_FETCH_MULTIPLIER", "5"))
VAULT_SEMANTIC_FETCH_MIN = int(os.environ.get("VAULT_SEMANTIC_FETCH_MIN", "200"))
