#!/usr/bin/env bash
set -euo pipefail

# ── Jambonz SIP Deployment ──
# One-command: installs Docker, starts Jambonz stack, configures account/app/SIP user,
# updates backend .env, prints Zoiper credentials.
# Usage: bash scripts/deploy_jambonz.sh [--skip-restart]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_DIR="$SCRIPT_DIR/.."
COMPOSE_DIR="$REPO_ROOT/docker/jambonz"
SKIP_RESTART=false
[[ "${1:-}" == "--skip-restart" ]] && SKIP_RESTART=true

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
fail()  { echo -e "${RED}[x]${NC} $*"; exit 1; }

# ── 1. Detect public IP ──
PUBLIC_IP="${PUBLIC_IP:-}"
if [ -z "$PUBLIC_IP" ]; then
    PUBLIC_IP=$(curl -s --max-time 5 https://ifconfig.me || hostname -I | awk '{print $1}')
fi
info "Public IP: $PUBLIC_IP"

# ── 2. Stop Asterisk if running (port 5060 conflict) ──
if pgrep -x asterisk &>/dev/null; then
    info "Stopping Asterisk (port 5060 needed by Jambonz)..."
    sudo systemctl stop asterisk 2>/dev/null || true
    sudo systemctl disable asterisk 2>/dev/null || true
    sudo pkill -x asterisk 2>/dev/null || true
    sleep 1
    if pgrep -x asterisk &>/dev/null; then
        sudo pkill -9 -x asterisk 2>/dev/null || true
        sleep 1
    fi
    info "Asterisk stopped"
fi

# ── 3. Check/install Docker ──
if ! command -v docker &>/dev/null; then
    info "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker "$USER" || true
    info "Docker installed. You may need to log out/in for group changes."
fi

if ! docker compose version &>/dev/null; then
    if ! docker-compose version &>/dev/null; then
        fail "Docker Compose not found. Install it: https://docs.docker.com/compose/install/"
    fi
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi
info "Docker OK, compose: $COMPOSE_CMD"

# ── 4. Create .env from template ──
if [ ! -f "$COMPOSE_DIR/.env" ]; then
    cp "$COMPOSE_DIR/.env.example" "$COMPOSE_DIR/.env"
fi
sed -i "s/^PUBLIC_IP=.*/PUBLIC_IP=$PUBLIC_IP/" "$COMPOSE_DIR/.env"
info "Updated PUBLIC_IP in $COMPOSE_DIR/.env"

# ── 5. Start Jambonz stack ──
info "Starting Jambonz stack..."
cd "$COMPOSE_DIR"
$COMPOSE_CMD up -d

# ── 6. Wait for API server healthy ──
info "Waiting for Jambonz API server..."
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://localhost:3000/v1/ServiceProviders 2>/dev/null || echo "000")
    # 401 means API is up (just needs auth), 200 means open
    if [ "$HTTP_CODE" = "401" ] || [ "$HTTP_CODE" = "200" ]; then
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo -n "."
done
echo ""

if [ $WAITED -ge $MAX_WAIT ]; then
    warn "Jambonz API not responding after ${MAX_WAIT}s. Check: $COMPOSE_CMD logs api-server"
    fail "Deployment failed"
fi
info "Jambonz API server ready (HTTP $HTTP_CODE)"

API="http://localhost:3000/v1"

# ── 7. Get admin API token from database ──
info "Reading admin API token from database..."
ADMIN_TOKEN=$($COMPOSE_CMD exec -T mysql mysql -ujambones -p"JambonzDB2026!" jambones \
    -N -e "SELECT token FROM api_keys WHERE account_sid IS NULL AND service_provider_sid IS NULL LIMIT 1" 2>/dev/null | tr -d '[:space:]')
if [ -z "$ADMIN_TOKEN" ]; then
    fail "Could not read admin API token from database"
fi
info "Admin token: ${ADMIN_TOKEN:0:8}..."

# Helper: authenticated curl (prints response, caller parses)
acurl() { curl -s -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" "$@"; }

# ── 8. Configure via REST API ──
SIP_REALM="voice.${PUBLIC_IP}.nip.io"
SIP_USER="${JAMBONZ_SIP_USER:-test}"

# 8a. Get or verify account (db-create makes a default one)
ACCOUNT_SID=$(acurl "$API/Accounts" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['account_sid'] if d else '')" 2>/dev/null || echo "")
if [ -z "$ACCOUNT_SID" ]; then
    fail "No account found. Check: $COMPOSE_CMD logs db-create"
fi

# Set SIP realm on the account
acurl -X PUT "$API/Accounts/$ACCOUNT_SID" -d "{\"sip_realm\": \"$SIP_REALM\"}" >/dev/null 2>&1
info "Account: $ACCOUNT_SID (realm: $SIP_REALM)"

# 8a2. Populate system_information (required for SIP realm routing)
SYS_INFO_EXISTS=$($COMPOSE_CMD exec -T mysql mysql -ujambones -p"JambonzDB2026!" jambones \
    -N -e "SELECT COUNT(*) FROM system_information" 2>/dev/null | tr -d '[:space:]')
if [ "$SYS_INFO_EXISTS" = "0" ] || [ -z "$SYS_INFO_EXISTS" ]; then
    $COMPOSE_CMD exec -T mysql mysql -ujambones -p"JambonzDB2026!" jambones \
        -e "INSERT INTO system_information (domain_name, sip_domain_name) VALUES ('$SIP_REALM', '$SIP_REALM')" 2>/dev/null
    info "System information: created (domain: $SIP_REALM)"
else
    $COMPOSE_CMD exec -T mysql mysql -ujambones -p"JambonzDB2026!" jambones \
        -e "UPDATE system_information SET domain_name='$SIP_REALM', sip_domain_name='$SIP_REALM'" 2>/dev/null
    info "System information: updated (domain: $SIP_REALM)"
fi

# 8a3. Populate sbc_addresses (required for SIP routing)
SBC_EXISTS=$($COMPOSE_CMD exec -T mysql mysql -ujambones -p"JambonzDB2026!" jambones \
    -N -e "SELECT COUNT(*) FROM sbc_addresses" 2>/dev/null | tr -d '[:space:]')
if [ "$SBC_EXISTS" = "0" ] || [ -z "$SBC_EXISTS" ]; then
    SBC_SID=$(python3 -c "import uuid; print(str(uuid.uuid4()))")
    $COMPOSE_CMD exec -T mysql mysql -ujambones -p"JambonzDB2026!" jambones \
        -e "INSERT INTO sbc_addresses (sbc_address_sid, ipv4, port, service_provider_sid) SELECT '$SBC_SID', '$PUBLIC_IP', 5060, service_provider_sid FROM service_providers LIMIT 1" 2>/dev/null
    info "SBC address: $PUBLIC_IP:5060"
else
    info "SBC address: already configured"
fi

# 8b. Create or get application
APP_SID=$(acurl "$API/Accounts/$ACCOUNT_SID/Applications" | python3 -c "
import sys, json
apps = json.load(sys.stdin)
for a in apps:
    if a.get('name') == 'voice-bot':
        print(a['application_sid'])
        sys.exit(0)
print('')
" 2>/dev/null || echo "")

if [ -z "$APP_SID" ]; then
    info "Creating voice-bot application..."
    APP_RESP=$(acurl -X POST "$API/Applications" \
        -d "{
            \"name\": \"voice-bot\",
            \"account_sid\": \"$ACCOUNT_SID\",
            \"call_hook\": {\"url\": \"ws://host.docker.internal:8000/ws/jambonz\", \"method\": \"POST\"},
            \"call_status_hook\": {\"url\": \"http://host.docker.internal:8000/api/jambonz/call-status\", \"method\": \"POST\"}
        }")
    APP_SID=$(echo "$APP_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('sid',''))" 2>/dev/null || echo "")
    if [ -z "$APP_SID" ]; then
        warn "Application creation response: $APP_RESP"
        fail "Could not create application"
    fi
    info "Application created: $APP_SID"
else
    # Update existing application webhook URLs
    acurl -X PUT "$API/Applications/$APP_SID" \
        -d "{
            \"call_hook\": {\"url\": \"ws://host.docker.internal:8000/ws/jambonz\", \"method\": \"POST\"},
            \"call_status_hook\": {\"url\": \"http://host.docker.internal:8000/api/jambonz/call-status\", \"method\": \"POST\"}
        }" >/dev/null 2>&1
    info "Application exists: $APP_SID (webhooks updated)"
fi

# 8b2. Link application to account for device calling (required for SIP phone routing)
$COMPOSE_CMD exec -T mysql mysql -ujambones -p"JambonzDB2026!" jambones \
    -e "UPDATE accounts SET device_calling_application_sid='$APP_SID' WHERE account_sid='$ACCOUNT_SID'" 2>/dev/null
info "Account linked to voice-bot application"

# 8c. Create or update SIP client (stored in MySQL `clients` table)
# Jambonz stores passwords encrypted with ENCRYPTION_SECRET (AES-256-CBC, JSON format)
ENV_FILE="$APP_DIR/.env"
EXISTING_PASS=$(grep '^JAMBONZ_SIP_PASSWORD=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- | tr -d "'" || echo "")
if [ -n "$EXISTING_PASS" ]; then
    SIP_PASSWORD="$EXISTING_PASS"
    info "SIP password: reusing existing from .env"
else
    SIP_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(16))")
    info "SIP password: generated new"
fi

# Encrypt password and write to MySQL directly via Node.js inside the sbc-sip-sidecar
# container. This ensures the SAME crypto code that decrypts also encrypts,
# and avoids shell escaping issues with the encrypted JSON.
UPSERT_RESULT=$($COMPOSE_CMD exec -T sbc-sip-sidecar node -e "
const crypto = require('crypto');
const mysql = require('mysql2/promise');
async function main() {
  const secretKey = crypto.createHash('sha256')
    .update(process.env.ENCRYPTION_SECRET)
    .digest('base64')
    .substring(0, 32);
  const iv = crypto.randomBytes(16);
  const cipher = crypto.createCipheriv('aes-256-cbc', secretKey, iv);
  let enc = cipher.update('$SIP_PASSWORD', 'utf8', 'hex');
  enc += cipher.final('hex');
  const encrypted = JSON.stringify({iv: iv.toString('hex'), content: enc});
  const conn = await mysql.createConnection({
    host: process.env.JAMBONES_MYSQL_HOST,
    user: process.env.JAMBONES_MYSQL_USER,
    password: process.env.JAMBONES_MYSQL_PASSWORD,
    database: process.env.JAMBONES_MYSQL_DATABASE
  });
  const [rows] = await conn.execute(
    'SELECT client_sid FROM clients WHERE account_sid = ? AND username = ?',
    ['$ACCOUNT_SID', '$SIP_USER']
  );
  if (rows.length > 0) {
    await conn.execute('UPDATE clients SET password = ? WHERE client_sid = ?', [encrypted, rows[0].client_sid]);
    console.log('updated');
  } else {
    const sid = crypto.randomUUID();
    await conn.execute(
      'INSERT INTO clients (client_sid, account_sid, username, password) VALUES (?, ?, ?, ?)',
      [sid, '$ACCOUNT_SID', '$SIP_USER', encrypted]
    );
    console.log('created');
  }
  await conn.end();
}
main().catch(e => { console.error(e.message); process.exit(1); });
" 2>/dev/null | tr -d '[:space:]')

if [ "$UPSERT_RESULT" = "created" ]; then
    info "SIP client created: $SIP_USER (encrypted)"
elif [ "$UPSERT_RESULT" = "updated" ]; then
    info "SIP client updated: $SIP_USER (encrypted)"
else
    fail "Could not create SIP client. Check sbc-sip-sidecar container."
fi

# 8d. Flush Redis and restart SBC services to pick up DB changes
# SBC services cache account/realm lookups at startup. If they started before
# the deploy script populated the DB, they have stale null entries.
info "Flushing Redis cache and restarting SBC services..."
$COMPOSE_CMD exec -T redis redis-cli FLUSHALL >/dev/null 2>&1
$COMPOSE_CMD restart sbc-sip-sidecar sbc-registrar >/dev/null 2>&1
sleep 5
info "SBC services restarted with fresh DB state"

# ── 9. Update backend .env ──
sed -i '/^SIP_ENABLED=/d; /^AUDIOSOCKET_/d; /^AMI_/d; /^JAMBONZ_/d; /^PUBLIC_IP=/d' "$ENV_FILE" 2>/dev/null || true

cat >> "$ENV_FILE" <<EOF

# ── Jambonz SIP Telephony (auto-configured by deploy_jambonz.sh) ──
JAMBONZ_ENABLED=true
JAMBONZ_API_BASE_URL=http://127.0.0.1:3000
JAMBONZ_ACCOUNT_SID=$ACCOUNT_SID
JAMBONZ_APP_SID=$APP_SID
JAMBONZ_SIP_REALM=$SIP_REALM
JAMBONZ_SIP_USER=$SIP_USER
JAMBONZ_SIP_PASSWORD=$SIP_PASSWORD
PUBLIC_IP=$PUBLIC_IP
EOF
info "Updated $ENV_FILE with Jambonz config"

# ── 10. Restart backend (unless --skip-restart) ──
if [ "$SKIP_RESTART" = true ]; then
    info "Skipping backend restart (--skip-restart)"
else
    if [ -f "$APP_DIR/scripts/restart_all.sh" ]; then
        info "Restarting backend..."
        bash "$APP_DIR/scripts/restart_all.sh" || warn "Backend restart returned non-zero"
    fi
fi

# ── 11. Verify ──
sleep 3
SIP_OK=false
if ss -ulnp 2>/dev/null | grep -q ':5060 ' || netstat -ulnp 2>/dev/null | grep -q ':5060 '; then
    SIP_OK=true
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Jambonz SIP Deployment Complete"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  SIP Port 5060:  $([ "$SIP_OK" = true ] && echo "OK" || echo "CHECKING...")"
echo "  Web Portal:     http://$PUBLIC_IP:3001"
echo "  API Server:     http://$PUBLIC_IP:3000"
echo "  Monitor Page:   http://$PUBLIC_IP:8000/sip_monitor.html"
echo ""
echo "  ── Zoiper Setup ──"
echo "  Server:    $SIP_REALM   (NOT the raw IP)"
echo "  Username:  $SIP_USER"
echo "  Password:  $SIP_PASSWORD"
echo "  Transport: UDP"
echo ""
echo "════════════════════════════════════════════════════════════"
