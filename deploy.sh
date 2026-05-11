#!/usr/bin/env bash
#
# Run from your Mac. Zips the project, scps to the VPS, runs install.sh remotely.
#
# Usage:    ./deploy.sh user@host
# Example:  ./deploy.sh root@45.32.123.45
#
# Re-running: idempotent — replaces files in ~/rover-nav on the VPS and restarts
# the service. Keeps the .env file untouched on subsequent runs (unless you pass
# fresh inputs via env vars).

set -euo pipefail

GREEN='\033[1;32m'; RED='\033[1;31m'; NC='\033[0m'
log() { echo -e "${GREEN}▶${NC} $*"; }
err() { echo -e "${RED}✗${NC} $*" >&2; }

[ -z "${1:-}" ] && { err "Usage: $0 user@host"; exit 1; }
SSH_TARGET=$1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# ── Prompt for config (only on first run) ───────────────────────────
read -rp "Domain (e.g. rover.alabs.cl): " DOMAIN
read -rp "OpenAI API key: " OPENAI_API_KEY
read -rsp "Access password: " ROVER_PASSWORD; echo
read -rp "Email for Let's Encrypt: " EMAIL
read -rp "Model [gpt-4o-mini]: " ROVER_MODEL
ROVER_MODEL=${ROVER_MODEL:-gpt-4o-mini}

# ── Build zip ───────────────────────────────────────────────────────
TMP=$(mktemp -d); trap "rm -rf $TMP" EXIT
ZIP="$TMP/rover-nav.zip"

cd "$SCRIPT_DIR"
log "Zipping project..."
zip -rq "$ZIP" . \
  -x "venv/*" "*/__pycache__/*" "*__pycache__*" \
     ".env" ".git/*" "*.zip" "deploy.sh" \
     "node_modules/*" ".DS_Store"
SIZE=$(du -h "$ZIP" | cut -f1)
log "Built archive ($SIZE)"

# ── Upload + remote install ─────────────────────────────────────────
log "Uploading to $SSH_TARGET..."
scp -q "$ZIP" "$SSH_TARGET":/tmp/rover-nav.zip

log "Running installer on remote..."
ssh -t "$SSH_TARGET" "bash -s" <<REMOTE
set -e
mkdir -p ~/rover-nav
cd ~/rover-nav
# Preserve existing .env if present
[ -f .env ] && cp .env /tmp/rover-nav.env.bak
unzip -oq /tmp/rover-nav.zip -d ~/rover-nav
[ -f /tmp/rover-nav.env.bak ] && mv /tmp/rover-nav.env.bak .env
rm /tmp/rover-nav.zip
chmod +x install.sh
DOMAIN='$DOMAIN' \
OPENAI_API_KEY='$OPENAI_API_KEY' \
ROVER_PASSWORD='$ROVER_PASSWORD' \
ROVER_MODEL='$ROVER_MODEL' \
EMAIL='$EMAIL' \
./install.sh
REMOTE

echo
log "✓ Deployed: https://$DOMAIN"
