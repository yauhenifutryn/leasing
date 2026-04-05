#!/usr/bin/env bash
set -euo pipefail

# Install and configure coturn TURN server for WebRTC media relay.
# Needed because cloud VMs are behind NAT; direct UDP doesn't work.
#
# Usage: sudo bash scripts/setup_turn.sh
# After running: TURN server listens on port 3478 (UDP+TCP).

PUBLIC_IP="${1:-$(curl -4 -s ifconfig.me)}"
TURN_USER="${TURN_USER:-voicebot}"
TURN_PASS="${TURN_PASS:-voicebot2026}"
TURN_PORT="${TURN_PORT:-3478}"
TURN_REALM="${TURN_REALM:-leasing.local}"

echo "[turn] Installing coturn..."
apt-get update -qq && apt-get install -y -qq coturn

echo "[turn] Configuring coturn..."
cat > /etc/turnserver.conf <<EOF
# coturn config for WebRTC media relay
listening-port=${TURN_PORT}
fingerprint
lt-cred-mech
realm=${TURN_REALM}
server-name=${TURN_REALM}
user=${TURN_USER}:${TURN_PASS}
external-ip=${PUBLIC_IP}
no-tls
no-dtls
no-cli
no-multicast-peers
# Restrict relay ports to a small range
min-port=49152
max-port=49252
EOF

# Enable coturn as a service
sed -i 's/^#TURNSERVER_ENABLED=1/TURNSERVER_ENABLED=1/' /etc/default/coturn 2>/dev/null || true

echo "[turn] Starting coturn..."
systemctl enable coturn
systemctl restart coturn

echo "[turn] Verifying..."
sleep 2
if systemctl is-active --quiet coturn; then
    echo "[turn] coturn running on ${PUBLIC_IP}:${TURN_PORT}"
    echo "[turn] User: ${TURN_USER}"
    echo "[turn] Done. WebRTC clients can now relay through this TURN server."
else
    echo "[turn] ERROR: coturn failed to start"
    journalctl -u coturn --no-pager -n 10
    exit 1
fi
