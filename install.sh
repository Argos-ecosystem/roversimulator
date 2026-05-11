#!/usr/bin/env bash
#
# Rover Navigator — installer for Ubuntu/Debian VPS with nginx + certbot.
#
# Runs IN the project directory after unzipping. Prompts for missing values
# unless passed as env vars: DOMAIN, OPENAI_API_KEY, ROVER_PASSWORD, EMAIL,
# ROVER_MODEL.
#
# Idempotent — re-running re-applies config and restarts the service.

set -euo pipefail

GREEN='\033[1;32m'; RED='\033[1;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}▶${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
err()  { echo -e "${RED}✗${NC} $*" >&2; }

[ "$(id -u)" -eq 0 ] && {
  err "Don't run this as root. Run as a normal sudo user.";
  exit 1;
}

INSTALL_DIR=$(pwd)
USER_NAME=$(whoami)
[ -f "$INSTALL_DIR/app.py" ] || { err "app.py not found in $INSTALL_DIR. Run from the unzipped project."; exit 1; }

# ── Inputs ──────────────────────────────────────────────────────────
DOMAIN=${DOMAIN:-}
OPENAI_API_KEY=${OPENAI_API_KEY:-}
ROVER_PASSWORD=${ROVER_PASSWORD:-}
ROVER_MODEL=${ROVER_MODEL:-gpt-4o-mini}
EMAIL=${EMAIL:-}

[ -z "$DOMAIN" ]         && read -rp "Domain (e.g. rover.alabs.cl): " DOMAIN
[ -z "$OPENAI_API_KEY" ] && read -rp "OpenAI API key: " OPENAI_API_KEY
[ -z "$ROVER_PASSWORD" ] && { read -rsp "Access password for the app: " ROVER_PASSWORD; echo; }
[ -z "$EMAIL" ]          && read -rp "Email for SSL cert (Let's Encrypt): " EMAIL

ROVER_SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')

# ── System packages ─────────────────────────────────────────────────
log "Installing system dependencies (idempotent)..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip nginx certbot python3-certbot-nginx unzip

# ── Python venv + libs ──────────────────────────────────────────────
log "Setting up Python venv..."
if [ ! -d "$INSTALL_DIR/venv" ]; then
  python3 -m venv "$INSTALL_DIR/venv"
fi
"$INSTALL_DIR/venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install --quiet flask openai gunicorn

# ── .env file (secrets) ─────────────────────────────────────────────
log "Writing .env file..."
umask 077
cat > "$INSTALL_DIR/.env" <<EOF
OPENAI_API_KEY=$OPENAI_API_KEY
ROVER_PASSWORD=$ROVER_PASSWORD
ROVER_SECRET=$ROVER_SECRET
ROVER_MODEL=$ROVER_MODEL
EOF
umask 022

# ── systemd service ─────────────────────────────────────────────────
log "Configuring systemd service..."
sudo tee /etc/systemd/system/rover.service > /dev/null <<EOF
[Unit]
Description=Rover Navigator
After=network.target

[Service]
User=$USER_NAME
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$INSTALL_DIR/venv/bin/gunicorn -w 1 -b 127.0.0.1:5050 --timeout 30 app:app
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rover --quiet
sudo systemctl restart rover

sleep 2
if ! sudo systemctl is-active --quiet rover; then
  err "Service failed to start. Check: sudo journalctl -u rover -n 50"
  exit 1
fi

# ── nginx vhost ─────────────────────────────────────────────────────
log "Configuring nginx vhost for $DOMAIN..."
sudo tee /etc/nginx/sites-available/$DOMAIN > /dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:5050;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 30s;
        client_max_body_size 5M;
    }
}
EOF

[ -L /etc/nginx/sites-enabled/$DOMAIN ] || sudo ln -s /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# ── SSL ─────────────────────────────────────────────────────────────
log "Requesting Let's Encrypt cert..."
if sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --email "$EMAIL" --redirect; then
  log "SSL ready."
else
  warn "Certbot failed. Check that $DOMAIN's DNS A-record points to this server. You can re-run later: sudo certbot --nginx -d $DOMAIN"
fi

# ── Done ────────────────────────────────────────────────────────────
echo
log "✓ Deployed."
echo "  URL:      https://$DOMAIN"
echo "  Password: (the one you set)"
echo "  Logs:     sudo journalctl -u rover -f"
echo "  Restart:  sudo systemctl restart rover"
echo "  Update:   re-run this script after replacing files in $INSTALL_DIR"
