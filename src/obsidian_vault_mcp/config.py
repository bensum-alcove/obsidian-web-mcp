import os
from pathlib import Path

# Vault configuration
VAULT_PATH = Path(os.environ.get("VAULT_PATH", os.path.expanduser("~/Obsidian/MyVault")))
VAULT_MCP_TOKEN = os.environ.get("VAULT_MCP_TOKEN", "")
TEAMBOT_MCP_TOKEN = os.environ.get("TEAMBOT_MCP_TOKEN", "")
VAULT_MCP_PORT = int(os.environ.get("VAULT_MCP_PORT", "8420"))

# OAuth 2.0 client credentials (for Claude app integration)
VAULT_OAUTH_CLIENT_ID = os.environ.get("VAULT_OAUTH_CLIENT_ID", "vault-mcp-client")
VAULT_OAUTH_CLIENT_SECRET = os.environ.get("VAULT_OAUTH_CLIENT_SECRET", "")

# Safety limits
MAX_CONTENT_SIZE = 1_000_000  # 1MB max write size
MAX_BATCH_SIZE = 20           # Max files per batch operation
MAX_SEARCH_RESULTS = 50       # Max results per search
DEFAULT_SEARCH_RESULTS = 20
MAX_LIST_DEPTH = 5            # Max directory recursion depth
CONTEXT_LINES = 2             # Default lines of context in search results

# Directories to never expose or modify
EXCLUDED_DIRS = {".obsidian", ".trash", ".git", ".DS_Store", ".semantic-index"}

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
RETRIEVAL_EXCLUDED_DIRS = EXCLUDED_DIRS | {SCRATCH_DIR_NAME}

# Frontmatter index refresh interval (seconds)
FRONTMATTER_INDEX_DEBOUNCE = 5.0

# hot.md ephemeral-cache budget in chars, frontmatter excluded. Single source of
# truth shared by scripts/hot-md-curate.py (enforcement) and scripts/dreaming.py
# (nightly report flag) -- see BS 2nd Brain/Alcove/Infrastructure/hot-md-structure.md.
HOT_MD_BUDGET_CHARS = 5000

# Rate limiting (requests per minute) -- track in-memory, enforce per-token
RATE_LIMIT_READ = 100
RATE_LIMIT_WRITE = 30

# vault_query keyword-leg tokenization (opt-out -- defaults enabled). When disabled,
# the keyword leg passes the raw question straight to ripgrep as it always has, byte-
# identical to pre-tokenization behaviour. Fastest revert path: env var edit, no git
# operation needed.
VAULT_QUERY_KEYWORD_TOKENIZE = os.environ.get("VAULT_QUERY_KEYWORD_TOKENIZE", "1") not in (
    "0", "false", "False", "",
)

# vault_search tokenized-augmentation (opt-out -- defaults enabled). Separate from
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

# vault_query RRF fusion sharpness (kill switch: env var edit, no git operation
# needed). Default 60 is byte-identical to pre-calibration behaviour. See
# tools/query.py's _rrf_fuse.
VAULT_QUERY_RRF_K = float(os.environ.get("VAULT_QUERY_RRF_K", "60"))

# vault_query canonical-state authority boost (kill switch: 1.0 = no-op, byte-identical
# to pre-calibration behaviour). See tools/query.py's _canonical_boost_factor.
VAULT_QUERY_CANONICAL_BOOST = float(os.environ.get("VAULT_QUERY_CANONICAL_BOOST", "1.0"))

# vault_query temporal decay: half-life in days, env-overridable.
# Longest matching path substring wins; unmatched paths use the default.
VAULT_QUERY_DEFAULT_HALF_LIFE_DAYS = float(os.environ.get("VAULT_QUERY_HALF_LIFE_DAYS", "90"))
VAULT_QUERY_HALF_LIFE_OVERRIDES = {
    "Claude-Code-Prompts/": float(os.environ.get("VAULT_QUERY_HALF_LIFE_CLAUDE_CODE_PROMPTS", "30")),
    "Skills/": float(os.environ.get("VAULT_QUERY_HALF_LIFE_SKILLS", "180")),
    "Clients/": float(os.environ.get("VAULT_QUERY_HALF_LIFE_CLIENTS", "365")),
}

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
VAULT_SEMANTIC_FETCH_MULTIPLIER = int(os.environ.get("VAULT_SEMANTIC_FETCH_MULTIPLIER", "5"))
VAULT_SEMANTIC_FETCH_MIN = int(os.environ.get("VAULT_SEMANTIC_FETCH_MIN", "100"))
