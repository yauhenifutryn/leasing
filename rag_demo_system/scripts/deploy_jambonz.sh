#!/usr/bin/env bash
set -euo pipefail

# ── Jambonz SIP Deployment ──
# One-command: installs Docker, starts Jambonz stack, configures account/app/SIP user,
# updates backend .env, restarts backend, prints Zoiper credentials.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_DIR="$SCRIPT_DIR/.."
COMPOSE_DIR="$REPO_ROOT/docker/jambonz"

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

# ── 3. Create .env from template ──
if [ ! -f "$COMPOSE_DIR/.env" ]; then
    cp "$COMPOSE_DIR/.env.example" "$COMPOSE_DIR/.env"
fi
sed -i "s/^PUBLIC_IP=.*/PUBLIC_IP=$PUBLIC_IP/" "$COMPOSE_DIR/.env"
info "Updated PUBLIC_IP in $COMPOSE_DIR/.env"

# ── 4. Start Jambonz stack ──
info "Starting Jambonz stack..."
cd "$COMPOSE_DIR"
$COMPOSE_CMD up -d

# ── 5. Wait for services healthy ──
info "Waiting for Jambonz services to be healthy..."
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s --max-time 3 http://localhost:3000/v1/ServiceProviders &>/dev/null; then
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
info "Jambonz API server ready"

API="http://localhost:3000/v1"

# ── 6. Get admin API token from database ──
info "Reading admin API token from database..."
ADMIN_TOKEN=$($COMPOSE_CMD exec -T mysql mysql -ujambones -p"JambonzDB2026!" jambones \
    -N -e "SELECT token FROM api_keys WHERE account_sid IS NULL AND service_provider_sid IS NULL LIMIT 1" 2>/dev/null | tr -d '[:space:]')
if [ -z "$ADMIN_TOKEN" ]; then
    fail "Could not read admin API token from database"
fi
info "Admin token: ${ADMIN_TOKEN:0:8}..."
AUTH="-H Authorization:\ Bearer\ $ADMIN_TOKEN"

# Helper: authenticated curl
acurl() { curl -s -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" "$@"; }

# ── 7. Configure via REST API ──
SIP_REALM="voice.${PUBLIC_IP}.nip.io"
SIP_USER="${JAMBONZ_SIP_USER:-test}"

# 7a. Get or verify account (db-create makes a default one)
ACCOUNT_SID=$(acurl "$API/Accounts" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['account_sid'] if d else '')" 2>/dev/null || echo "")
if [ -z "$ACCOUNT_SID" ]; then
    fail "No account found. Check: docker compose logs db-create"
fi

# Set SIP realm on the account
acurl -X PUT "$API/Accounts/$ACCOUNT_SID" \
    -d "{\"sip_realm\": \"$SIP_REALM\"}" >/dev/null 2>&1
info "Account: $ACCOUNT_SID (realm: $SIP_REALM)"

# 7b. Create or get application (WebSocket)
APP_SID=$(acurl "$API/Accounts/$ACCOUNT_SID/Applications" | python3 -c "
import sys, json
apps = json.load(sys.stdin)
# Find existing voice-bot app or return empty
for a in apps:
    if a.get('name') == 'voice-bot':
        print(a['application_sid'])
        sys.exit(0)
print('')
" 2>/dev/null || echo "")

if [ -z "$APP_SID" ]; then
    APP_SID=$(acurl -X POST "$API/Applications" \
        -d "{
            \"name\": \"voice-bot\",
            \"account_sid\": \"$ACCOUNT_SID\",
            \"call_hook\": {\"url\": \"ws://host.docker.internal:8000/ws/jambonz\", \"method\": \"POST\"},
            \"call_status_hook\": {\"url\": \"http://host.docker.internal:8000/api/jambonz/call-status\", \"method\": \"POST\"}
        }" | python3 -c "import sys,json; print(json.load(sys.stdin).get('sid',''))" 2>/dev/null || echo "")
fi
if [ -z "$APP_SID" ]; then
    fail "Could not create application. Check: acurl $API/Accounts/$ACCOUNT_SID/Applications"
fi
info "Application: $APP_SID"

# 7c. Create or update SIP credentials
# Generate password only if not already in backend .env
ENV_FILE="$APP_DIR/.env"
EXISTING_PASS=$(grep '^JAMBONZ_SIP_PASSWORD=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- | tr -d "'" || echo "")
if [ -n "$EXISTING_PASS" ]; then
    SIP_PASSWORD="$EXISTING_PASS"
    info "SIP password: reusing existing from .env"
else
    SIP_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(16))")
    info "SIP password: generated new"
fi

# Check if SIP user exists
SIP_EXISTS=$(acurl "$API/Accounts/$ACCOUNT_SID/SipCredentials" | python3 -c "
import sys, json
creds = json.load(sys.stdin)
for c in creds:
    if c.get('username') == '$SIP_USER':
        print(c.get('sip_credential_sid', 'exists'))
        sys.exit(0)
print('')
" 2>/dev/null || echo "")

if [ -n "$SIP_EXISTS" ] && [ "$SIP_EXISTS" != "exists" ]; then
    # Update existing credentials
    acurl -X PUT "$API/Accounts/$ACCOUNT_SID/SipCredentials/$SIP_EXISTS" \
        -d "{\"password\": \"$SIP_PASSWORD\"}" >/dev/null 2>&1
    info "SIP user updated: $SIP_USER"
else
    # Create new
    acurl -X POST "$API/Accounts/$ACCOUNT_SID/SipCredentials" \
        -d "{\"username\": \"$SIP_USER\", \"password\": \"$SIP_PASSWORD\"}" >/dev/null 2>&1
    info "SIP user created: $SIP_USER"
fi

# ── 8. Update backend .env ──
# Remove old SIP/Jambonz vars
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

# ── 9. Restart backend ──
if [ -f "$APP_DIR/scripts/restart_all.sh" ]; then
    info "Restarting backend..."
    bash "$APP_DIR/scripts/restart_all.sh" || warn "Backend restart returned non-zero"
fi

# ── 10. Verify ──
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
echo "  Server:    $PUBLIC_IP"
echo "  Username:  $SIP_USER"
echo "  Password:  $SIP_PASSWORD"
echo "  Transport: UDP"
echo ""
echo "════════════════════════════════════════════════════════════"
