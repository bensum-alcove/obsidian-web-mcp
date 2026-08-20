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

# Frontmatter index refresh interval (seconds)
FRONTMATTER_INDEX_DEBOUNCE = 5.0

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
# no git operation needed). Default 60 is byte-identical to pre-calibration behaviour.
# vault-query-calibration-v2 diagnosis: k=60 is flat enough, relative to the ~150-
# candidate fetch depth, that a document appearing at a middling rank in BOTH legs
# routinely out-scores a document ranked #1 in only ONE leg (their summed 1/(k+rank)
# terms exceed the single leg's 1/(k+1)). This was the single highest-leverage,
# most-repeating failure mode in the v3 baseline diagnosis -- see this build's output
# doc for the per-question rank trace. Lowering k sharpens the top of the curve so a
# confident single-leg #1 is harder to displace, without changing the behaviour for
# genuine both-legs-agree consensus (which wins at any k>0 since it's always ~2x a
# single leg's score at the same rank).
VAULT_QUERY_RRF_K = float(os.environ.get("VAULT_QUERY_RRF_K", "60"))

# vault_query canonical-state authority boost (kill switch: 1.0 = no-op, byte-identical
# to pre-calibration behaviour). Multiplies the fused score of any result whose
# frontmatter declares `type: canonical-state` before temporal decay is applied. The
# vault's own documented precedence rule (infrastructure.md's "Current-state authority
# note") says these records are the current-state authority over general prose --
# this mechanism is a modest, multiplicative nudge toward that rule, not a hard
# override, so a genuinely stronger match can still win. Inert (no matching files) in
# any vault that has no Canonical State/records/ tree yet, e.g. CB/Alcove Brain at the
# time of this build.
VAULT_QUERY_CANONICAL_BOOST = float(os.environ.get("VAULT_QUERY_CANONICAL_BOOST", "1.0"))

# vault_query temporal decay: half-life in days, env-overridable.
# Longest matching path substring wins; unmatched paths use the default.
VAULT_QUERY_DEFAULT_HALF_LIFE_DAYS = float(os.environ.get("VAULT_QUERY_HALF_LIFE_DAYS", "90"))
VAULT_QUERY_HALF_LIFE_OVERRIDES = {
    "Claude-Code-Prompts/": float(os.environ.get("VAULT_QUERY_HALF_LIFE_CLAUDE_CODE_PROMPTS", "30")),
    "Skills/": float(os.environ.get("VAULT_QUERY_HALF_LIFE_SKILLS", "180")),
    "Clients/": float(os.environ.get("VAULT_QUERY_HALF_LIFE_CLIENTS", "365")),
}
