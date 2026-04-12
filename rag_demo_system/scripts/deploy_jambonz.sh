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
if command -v asterisk &>/dev/null || ss -ulnp 2>/dev/null | grep -q ':5060 '; then
    info "Stopping Asterisk (port 5060 needed by Jambonz)..."
    sudo systemctl stop asterisk 2>/dev/null || true
    sudo systemctl disable asterisk 2>/dev/null || true
    # Kill any remaining asterisk process
    sudo pkill -f asterisk 2>/dev/null || true
    sleep 1
    if ss -ulnp 2>/dev/null | grep -q ':5060 '; then
        fail "Port 5060 still in use after stopping Asterisk. Check: sudo ss -ulnp | grep 5060"
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

# ── 6. Configure via REST API ──
# 6a. Create service provider
SP_SID=$(curl -s -X POST "$API/ServiceProviders" \
    -H "Content-Type: application/json" \
    -d '{"name":"voice-pipeline"}' | python3 -c "import sys,json; print(json.load(sys.stdin).get('sid',''))" 2>/dev/null || echo "")

if [ -z "$SP_SID" ]; then
    SP_SID=$(curl -s "$API/ServiceProviders" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['service_provider_sid'] if d else '')" 2>/dev/null || echo "")
fi
info "Service Provider: $SP_SID"

# 6b. Create account
SIP_REALM="voice.${PUBLIC_IP}.nip.io"
SIP_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(16))")

ACCOUNT_SID=$(curl -s -X POST "$API/Accounts" \
    -H "Content-Type: application/json" \
    -d "{
        \"name\": \"leasing-voice\",
        \"service_provider_sid\": \"$SP_SID\",
        \"sip_realm\": \"$SIP_REALM\",
        \"webhook_secret\": \"$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')\"
    }" | python3 -c "import sys,json; print(json.load(sys.stdin).get('sid',''))" 2>/dev/null || echo "")

if [ -z "$ACCOUNT_SID" ]; then
    ACCOUNT_SID=$(curl -s "$API/Accounts" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['account_sid'] if d else '')" 2>/dev/null || echo "")
fi
info "Account: $ACCOUNT_SID (realm: $SIP_REALM)"

# 6c. Create application (WebSocket)
APP_WS_URL="ws://host.docker.internal:8000/ws/jambonz"
APP_SID=$(curl -s -X POST "$API/Accounts/$ACCOUNT_SID/Applications" \
    -H "Content-Type: application/json" \
    -d "{
        \"name\": \"voice-bot\",
        \"account_sid\": \"$ACCOUNT_SID\",
        \"call_hook\": {\"url\": \"$APP_WS_URL\", \"method\": \"WS\"},
        \"call_status_hook\": {\"url\": \"http://host.docker.internal:8000/api/jambonz/call-status\", \"method\": \"POST\"},
        \"speech_synthesis_vendor\": \"custom\",
        \"speech_recognizer_vendor\": \"custom\"
    }" | python3 -c "import sys,json; print(json.load(sys.stdin).get('sid',''))" 2>/dev/null || echo "")

if [ -z "$APP_SID" ]; then
    APP_SID=$(curl -s "$API/Accounts/$ACCOUNT_SID/Applications" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['application_sid'] if d else '')" 2>/dev/null || echo "")
fi
info "Application: $APP_SID"

# 6d. Create SIP user
SIP_USER="${JAMBONZ_SIP_USER:-test}"
curl -s -X POST "$API/Accounts/$ACCOUNT_SID/SipCredentials" \
    -H "Content-Type: application/json" \
    -d "{
        \"username\": \"$SIP_USER\",
        \"password\": \"$SIP_PASSWORD\"
    }" >/dev/null 2>&1 || true
info "SIP user created: $SIP_USER"

# ── 7. Update backend .env ──
ENV_FILE="$APP_DIR/.env"
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

# ── 8. Restart backend ──
if [ -f "$APP_DIR/scripts/restart_all.sh" ]; then
    info "Restarting backend..."
    bash "$APP_DIR/scripts/restart_all.sh" || warn "Backend restart returned non-zero"
fi

# ── 9. Verify ──
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
