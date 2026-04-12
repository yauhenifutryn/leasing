#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# diagnose_sip.sh -- Comprehensive SIP diagnostic
# Run on the server to check every layer of the SIP stack.
# ---------------------------------------------------------------------------
set -uo pipefail

APP_DIR="${APP_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

log() { echo "[diag] $*"; }
pass() { echo "[diag][OK]   $*"; }
fail() { echo "[diag][FAIL] $*"; }
warn() { echo "[diag][WARN] $*"; }

log "============================================="
log "  SIP Diagnostic"
log "============================================="

# ---------------------------------------------------------------------------
# 1. Asterisk process
# ---------------------------------------------------------------------------
log ""
log "--- 1. Asterisk process ---"
if pgrep -x asterisk > /dev/null; then
    pass "Asterisk process running (PID: $(pgrep -x asterisk | head -1))"
else
    fail "Asterisk process NOT running"
fi

ASTERISK_VER=$(asterisk -V 2>/dev/null || echo "NOT FOUND")
log "  Version: $ASTERISK_VER"

# ---------------------------------------------------------------------------
# 2. Ports
# ---------------------------------------------------------------------------
log ""
log "--- 2. Network ports ---"

if ss -ulnp 2>/dev/null | grep -q ":5060"; then
    pass "SIP UDP port 5060 listening"
else
    fail "SIP UDP port 5060 NOT listening"
fi

if ss -tlnp 2>/dev/null | grep -q ":9092"; then
    pass "AudioSocket TCP port 9092 listening"
else
    fail "AudioSocket TCP port 9092 NOT listening"
fi

if ss -tlnp 2>/dev/null | grep -q ":8000"; then
    pass "Web app port 8000 listening"
else
    fail "Web app port 8000 NOT listening"
fi

# ---------------------------------------------------------------------------
# 3. Asterisk modules
# ---------------------------------------------------------------------------
log ""
log "--- 3. Asterisk modules ---"

for mod in app_audiosocket res_pjsip; do
    if asterisk -rx "module show like $mod" 2>/dev/null | grep -q "$mod"; then
        pass "Module $mod loaded"
    else
        fail "Module $mod NOT loaded"
    fi
done

# ---------------------------------------------------------------------------
# 4. PJSIP endpoints
# ---------------------------------------------------------------------------
log ""
log "--- 4. PJSIP endpoints ---"
ENDPOINTS=$(asterisk -rx "pjsip show endpoints" 2>/dev/null)
echo "$ENDPOINTS"

# ---------------------------------------------------------------------------
# 5. PJSIP config check
# ---------------------------------------------------------------------------
log ""
log "--- 5. PJSIP config ---"
log "  Endpoint context:"
grep "^context" /etc/asterisk/pjsip.conf 2>/dev/null || warn "No context found in pjsip.conf"

log "  Auth sections:"
grep -A2 "type = auth" /etc/asterisk/pjsip.conf 2>/dev/null || warn "No auth sections found"

# ---------------------------------------------------------------------------
# 6. Dialplan
# ---------------------------------------------------------------------------
log ""
log "--- 6. Dialplan ---"
log "  All contexts:"
asterisk -rx "dialplan show" 2>/dev/null | grep "Context" | head -10

log ""
log "  voice-bot context:"
asterisk -rx "dialplan show voice-bot" 2>/dev/null || warn "voice-bot context not found"

log ""
log "  default context:"
asterisk -rx "dialplan show default" 2>/dev/null | head -10

# ---------------------------------------------------------------------------
# 7. Competing config files
# ---------------------------------------------------------------------------
log ""
log "--- 7. Config file check ---"
for f in extensions.lua extensions.ael; do
    if [ -f "/etc/asterisk/$f" ]; then
        fail "/etc/asterisk/$f EXISTS (may override extensions.conf)"
    else
        pass "/etc/asterisk/$f absent or backed up"
    fi
done

# ---------------------------------------------------------------------------
# 8. Direct AudioSocket test
# ---------------------------------------------------------------------------
log ""
log "--- 8. Direct AudioSocket TCP test ---"
# Try connecting to AudioSocket port and send a UUID frame
AUDIOSOCKET_TEST=$(python3 -c "
import socket, struct, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3)
try:
    s.connect(('127.0.0.1', 9092))
    # Send UUID frame
    uuid_str = b'test-diag-0000-0000-000000000000'
    header = struct.pack('!BH', 0x01, len(uuid_str))
    s.sendall(header + uuid_str)
    # Try to read response (may get audio back from TTS intro)
    s.settimeout(5)
    try:
        data = s.recv(1024)
        print(f'CONNECTED: received {len(data)} bytes back')
    except socket.timeout:
        print('CONNECTED: no response in 5s (app may not be processing)')
    s.close()
except ConnectionRefusedError:
    print('REFUSED: port 9092 not accepting connections')
except socket.timeout:
    print('TIMEOUT: connection timed out')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
log "  $AUDIOSOCKET_TEST"

# ---------------------------------------------------------------------------
# 9. SIP env and app config
# ---------------------------------------------------------------------------
log ""
log "--- 9. App configuration ---"
if [ -f "$APP_DIR/.env" ]; then
    SIP_ENABLED=$(grep "^SIP_ENABLED" "$APP_DIR/.env" 2>/dev/null || echo "NOT SET")
    AMI_SECRET=$(grep "^AMI_SECRET" "$APP_DIR/.env" 2>/dev/null | sed 's/AMI_SECRET=//' || echo "NOT SET")
    log "  SIP_ENABLED: $SIP_ENABLED"
    log "  AMI_SECRET: ${AMI_SECRET:0:8}... (truncated)"
else
    fail ".env file not found"
fi

# ---------------------------------------------------------------------------
# 10. Recent Asterisk logs
# ---------------------------------------------------------------------------
log ""
log "--- 10. Recent Asterisk logs (last 20 relevant lines) ---"
grep -E "REGISTER|INVITE|AudioSocket|UUID|voice-bot|error|ERROR" /var/log/asterisk/messages 2>/dev/null | tail -20

# ---------------------------------------------------------------------------
# 11. Test INVITE simulation
# ---------------------------------------------------------------------------
log ""
log "--- 11. SIP INVITE simulation ---"
log "  Enabling verbose logging..."
asterisk -rx "core set verbose 10" > /dev/null 2>&1

log "  Sending test INVITE via Asterisk CLI..."
# Use Asterisk to originate a test call to extension 100
asterisk -rx "channel originate PJSIP/100@dev application Wait 1" 2>/dev/null &
ORIG_PID=$!
sleep 3
kill $ORIG_PID 2>/dev/null || true

log "  Checking logs for call trace..."
tail -30 /var/log/asterisk/messages | grep -v "declined to load" | grep -v "chan_sip" | tail -15

asterisk -rx "core set verbose 0" > /dev/null 2>&1

log ""
log "============================================="
log "  Diagnostic complete"
log "============================================="
