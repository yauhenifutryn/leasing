#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# deploy_sip.sh -- One-command SIP deployment on top of existing voice stack
#
# Run AFTER provision_server.sh and smoke_test.sh have passed.
# This script: installs Asterisk, configures SIP, enables SIP in .env,
# restarts the app, and verifies SIP is ready for Zoiper connections.
#
# Usage:
#   bash scripts/deploy_sip.sh
# ---------------------------------------------------------------------------
set -euo pipefail

APP_DIR="${APP_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$APP_DIR"

log() { echo "[sip-deploy] $*"; }
fail() { echo "[sip-deploy][FAIL] $*"; exit 1; }
pass() { echo "[sip-deploy][OK]   $*"; }

log "============================================="
log "  SIP Telephony Deployment"
log "  App dir: $APP_DIR"
log "============================================="

# ---------------------------------------------------------------------------
# Step 1: Install Asterisk
# ---------------------------------------------------------------------------
log ""
log "--- Step 1: Install Asterisk ---"

if command -v asterisk &>/dev/null; then
    log "Asterisk already installed: $(asterisk -V 2>/dev/null || echo 'unknown version')"
else
    log "Installing Asterisk..."
    apt-get update -qq
    apt-get install -y -qq asterisk > /dev/null 2>&1 || fail "Asterisk installation failed"
    pass "Asterisk installed: $(asterisk -V 2>/dev/null || echo 'unknown version')"
fi

# Verify AudioSocket module
if asterisk -rx "module show like audiosocket" 2>/dev/null | grep -q audiosocket; then
    pass "app_audiosocket module available"
else
    log "WARNING: app_audiosocket not found. AudioSocket may not work."
    log "  Need Asterisk 16+ for AudioSocket, 20.14+ for DTMF."
fi

# ---------------------------------------------------------------------------
# Step 2: Configure Asterisk
# ---------------------------------------------------------------------------
log ""
log "--- Step 2: Configure Asterisk ---"

ASTERISK_ETC="/etc/asterisk"

# Generate secure passwords
DEV_PASS=$(openssl rand -hex 12)
CLIENT_PASS=$(openssl rand -hex 12)
AMI_PASS=$(openssl rand -hex 12)

# Copy configs and inject passwords
cp "$APP_DIR/config/asterisk/pjsip.conf" "$ASTERISK_ETC/pjsip.conf"
cp "$APP_DIR/config/asterisk/extensions.conf" "$ASTERISK_ETC/extensions.conf"
cp "$APP_DIR/config/asterisk/manager.conf" "$ASTERISK_ETC/manager.conf"

sed -i "s/CHANGE_ME_dev_password/$DEV_PASS/" "$ASTERISK_ETC/pjsip.conf"
sed -i "s/CHANGE_ME_client_password/$CLIENT_PASS/" "$ASTERISK_ETC/pjsip.conf"
sed -i "s/CHANGE_ME_ami_secret/$AMI_PASS/" "$ASTERISK_ETC/manager.conf"

pass "Asterisk configs installed with generated passwords"

# ---------------------------------------------------------------------------
# Step 3: Start Asterisk
# ---------------------------------------------------------------------------
log ""
log "--- Step 3: Start Asterisk ---"

# Stop if running, then start fresh
systemctl stop asterisk 2>/dev/null || true
sleep 1
systemctl enable asterisk 2>/dev/null || true
systemctl start asterisk 2>/dev/null || {
    # Fallback: start directly if systemctl is unavailable (containers)
    asterisk -C /etc/asterisk/asterisk.conf &
    sleep 2
}

# Verify Asterisk is running
if asterisk -rx "core show version" &>/dev/null; then
    pass "Asterisk running"
else
    fail "Asterisk failed to start"
fi

# Reload PJSIP config
asterisk -rx "module reload res_pjsip.so" 2>/dev/null || true
asterisk -rx "dialplan reload" 2>/dev/null || true
sleep 1

# Verify endpoints
ENDPOINTS=$(asterisk -rx "pjsip show endpoints" 2>/dev/null || echo "")
if echo "$ENDPOINTS" | grep -q "dev-user"; then
    pass "SIP endpoint dev-user registered"
else
    log "WARNING: dev-user endpoint not visible yet (may need Zoiper to register)"
fi

# ---------------------------------------------------------------------------
# Step 4: Enable SIP in .env
# ---------------------------------------------------------------------------
log ""
log "--- Step 4: Enable SIP in .env ---"

ENV_FILE="$APP_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    fail ".env file not found at $ENV_FILE. Run provision_server.sh first."
fi

# Remove old SIP settings if any, then append fresh ones
sed -i '/^SIP_ENABLED=/d; /^AUDIOSOCKET_/d; /^AMI_/d' "$ENV_FILE"
cat >> "$ENV_FILE" << SIPEOF

# ── SIP Telephony (added by deploy_sip.sh) ──
SIP_ENABLED=true
AUDIOSOCKET_HOST=127.0.0.1
AUDIOSOCKET_PORT=9092
AMI_HOST=127.0.0.1
AMI_PORT=5038
AMI_USERNAME=voicebot
AMI_SECRET=$AMI_PASS
SIPEOF

pass "SIP enabled in .env"

# ---------------------------------------------------------------------------
# Step 5: Restart the voice app
# ---------------------------------------------------------------------------
log ""
log "--- Step 5: Restart voice app ---"

SUPERVISORCTL="$APP_DIR/.venv/bin/supervisorctl"
SUPERVISOR_CONF="$APP_DIR/scripts/supervisord.conf"

if [ -f "$SUPERVISORCTL" ] && [ -f "$SUPERVISOR_CONF" ]; then
    "$SUPERVISORCTL" -c "$SUPERVISOR_CONF" restart backend 2>/dev/null || {
        log "supervisorctl restart failed, trying full stack restart..."
        bash "$APP_DIR/scripts/restart_all.sh" 2>/dev/null || true
    }
else
    log "No supervisor found, attempting restart_all.sh..."
    bash "$APP_DIR/scripts/restart_all.sh" 2>/dev/null || true
fi

log "Waiting 10s for app to start..."
sleep 10

# ---------------------------------------------------------------------------
# Step 6: Verify SIP is ready
# ---------------------------------------------------------------------------
log ""
log "--- Step 6: Verify SIP ---"

# Check app is responding
APP_STATUS=$(curl -s --max-time 5 http://127.0.0.1:8000/api/voice/status 2>/dev/null || echo "")
if echo "$APP_STATUS" | grep -q '"ok":true'; then
    pass "Voice app responding"
else
    log "WARNING: Voice app not responding yet. It may still be loading."
fi

# Check AudioSocket port is listening
if ss -tlnp 2>/dev/null | grep -q ":9092"; then
    pass "AudioSocket server listening on port 9092"
elif netstat -tlnp 2>/dev/null | grep -q ":9092"; then
    pass "AudioSocket server listening on port 9092"
else
    log "WARNING: AudioSocket port 9092 not detected yet. Check app logs."
fi

# Check SIP port
if ss -ulnp 2>/dev/null | grep -q ":5060"; then
    pass "Asterisk SIP listening on UDP port 5060"
elif netstat -ulnp 2>/dev/null | grep -q ":5060"; then
    pass "Asterisk SIP listening on UDP port 5060"
else
    log "WARNING: SIP port 5060 not detected"
fi

# Check monitor page
MONITOR_STATUS=$(curl -s --max-time 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/sip_monitor.html 2>/dev/null || echo "000")
if [ "$MONITOR_STATUS" = "200" ]; then
    pass "Monitor page accessible at /sip_monitor.html"
else
    log "WARNING: Monitor page returned HTTP $MONITOR_STATUS"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "<SERVER_IP>")

log ""
log "============================================="
log "  SIP DEPLOYMENT COMPLETE"
log "============================================="
log ""
log "  Zoiper credentials:"
log "    Domain:   $SERVER_IP"
log "    Username: dev"
log "    Password: $DEV_PASS"
log ""
log "    Client account:"
log "    Username: client"
log "    Password: $CLIENT_PASS"
log ""
log "  Monitor page:"
log "    http://$SERVER_IP:8000/sip_monitor.html"
log ""
log "  Dial 100 from Zoiper to reach the voice bot."
log "============================================="
