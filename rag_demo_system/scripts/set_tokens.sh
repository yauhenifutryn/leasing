#!/usr/bin/env bash
set -euo pipefail

# set_tokens.sh
#
# Patches tool-use credentials (Calculator + SMS + CRM) into .env from the
# current shell env, then restarts ONLY the backend process (vLLM, Whisper,
# Silero stay up — no expensive model reload).
#
# Use this after provision when you forgot to export tokens beforehand.
# If you exported BEFORE provision, provision already put them in .env and
# you don't need this script.
#
# Usage:
#   export CALCULATOR_API_TOKEN='...'
#   export SMS_API_LOGIN='...' SMS_API_PASSWORD='...'
#   bash scripts/set_tokens.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$APP_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "[set_tokens] ERROR: $ENV_FILE does not exist. Run provision_server.sh first."
  exit 1
fi

# Upsert a single KEY=VALUE pair in .env. Value is wrapped in single quotes.
# If the key is present, the existing line is replaced; otherwise appended.
_upsert() {
  local key="$1"
  local value="$2"
  # Escape single quotes inside the value for safe .env quoting.
  local escaped="${value//\'/\'\\\'\'}"
  local new_line="${key}='${escaped}'"
  if grep -qE "^${key}=" "$ENV_FILE"; then
    # sed -i portability: use a temp file to avoid GNU/BSD differences.
    local tmp
    tmp="$(mktemp)"
    awk -v k="$key" -v v="$new_line" '
      BEGIN { re = "^" k "=" }
      {
        if ($0 ~ re) { print v; next }
        print
      }
    ' "$ENV_FILE" > "$tmp"
    mv "$tmp" "$ENV_FILE"
  else
    printf '%s\n' "$new_line" >> "$ENV_FILE"
  fi
}

_updated=()
for key in CALCULATOR_API_BASE_URL CALCULATOR_API_TOKEN USD_BYN_RATE \
           SMS_API_LOGIN SMS_API_PASSWORD SMS_SENDER_NAME \
           CRM_WEBHOOK_URL CRM_WEBHOOK_TOKEN; do
  # Only upsert when the env var is actually set (non-empty).
  if [ -n "${!key:-}" ]; then
    _upsert "$key" "${!key}"
    _updated+=("$key")
  fi
done

if [ "${#_updated[@]}" -eq 0 ]; then
  echo "[set_tokens] No tool env vars found in shell. Nothing to update."
  echo "[set_tokens] Export them first, e.g.:"
  echo "[set_tokens]   export CALCULATOR_API_TOKEN='...'"
  echo "[set_tokens]   export SMS_API_LOGIN='...' SMS_API_PASSWORD='...'"
  exit 0
fi

echo "[set_tokens] Patched .env: ${_updated[*]}"

# Restart backend only — vLLM, Whisper, Silero keep running.
SUPERVISORCTL="$APP_DIR/.venv/bin/supervisorctl"
SUPERVISOR_CONF="$APP_DIR/scripts/supervisord.conf"
if [ -x "$SUPERVISORCTL" ] && [ -f "$SUPERVISOR_CONF" ]; then
  echo "[set_tokens] Restarting backend..."
  "$SUPERVISORCTL" -c "$SUPERVISOR_CONF" restart backend 2>/dev/null \
    && echo "[set_tokens] Backend restarted." \
    || echo "[set_tokens] WARNING: supervisorctl restart failed. Start supervisord first: bash scripts/stack.sh up"
else
  echo "[set_tokens] WARNING: supervisorctl not found at $SUPERVISORCTL."
  echo "[set_tokens] Start the stack manually: bash scripts/stack.sh up"
fi

echo "[set_tokens] Done. Next: bash scripts/smoke_test.sh"
