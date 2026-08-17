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
EXCLUDED_DIRS = {".obsidian", ".trash", ".git", ".DS_Store", ".semantic-index"}

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

# vault_query temporal decay: half-life in days, env-overridable.
# Longest matching path substring wins; unmatched paths use the default.
VAULT_QUERY_DEFAULT_HALF_LIFE_DAYS = float(os.environ.get("VAULT_QUERY_HALF_LIFE_DAYS", "90"))
VAULT_QUERY_HALF_LIFE_OVERRIDES = {
    "Claude-Code-Prompts/": float(os.environ.get("VAULT_QUERY_HALF_LIFE_CLAUDE_CODE_PROMPTS", "30")),
    "Skills/": float(os.environ.get("VAULT_QUERY_HALF_LIFE_SKILLS", "180")),
    "Clients/": float(os.environ.get("VAULT_QUERY_HALF_LIFE_CLIENTS", "365")),
}
