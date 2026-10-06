#!/usr/bin/env bash
# Read-only deploy verification for obsidian-web-mcp: confirms this checkout's
# HEAD has landed on origin/main, and that every vault-mcp instance this repo
# serves (BS Brain, CB Brain, Alcove Brain) is RUNNING with a healthy /health
# endpoint. Never restarts anything and never requires the running process
# SHA to equal HEAD -- that is sha_deployed's job where a build declares it.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SUPERVISORCTL="${SUPERVISORCTL:-$HOME/.local/bin/supervisorctl}"

# program -> the local port its /health endpoint listens on (brains.conf)
PROGRAMS=(bs-brain-vault cb-brain-vault alcove-brain-vault)
declare -A HEALTH_PORT=(
  [bs-brain-vault]=8420
  [cb-brain-vault]=8423
  [alcove-brain-vault]=8426
)

fail() {
  echo "FAIL: $1"
  exit 1
}

# 1. This checkout's HEAD must be origin/main, or already merged into it.
git -C "$REPO" fetch origin --quiet 2>/dev/null || fail "git fetch origin failed"
head_sha=$(git -C "$REPO" rev-parse HEAD)
main_sha=$(git -C "$REPO" rev-parse origin/main)
if [ "$head_sha" != "$main_sha" ] && ! git -C "$REPO" merge-base --is-ancestor HEAD origin/main; then
  fail "HEAD ($head_sha) is not origin/main ($main_sha) or an ancestor of it"
fi

# 2. Every serving supervisor program must be RUNNING.
for prog in "${PROGRAMS[@]}"; do
  status_line=$("$SUPERVISORCTL" status "$prog" 2>&1) || fail "supervisorctl status $prog failed: $status_line"
  read -r _ state _ <<<"$status_line"
  [ "$state" = "RUNNING" ] || fail "$prog is not RUNNING: $status_line"
done

# 3. Each program's own /health endpoint must answer OK (no auth, no secrets).
for prog in "${PROGRAMS[@]}"; do
  port="${HEALTH_PORT[$prog]}"
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${port}/health" || echo "000")
  [ "$code" = "200" ] || fail "$prog /health on port $port returned $code (expected 200)"
done

echo "PASS: HEAD on origin/main; bs-brain-vault/cb-brain-vault/alcove-brain-vault RUNNING; /health 200 on 8420/8423/8426"
