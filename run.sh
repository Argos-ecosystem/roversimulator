#!/usr/bin/env bash
#
# Local dev runner — set up venv, install deps, run Flask dev server.
# Reads OPENAI_API_KEY from .env if present, otherwise from the shell.

set -euo pipefail

cd "$(dirname "$0")"

# Load .env if it exists
if [ -f .env ]; then
  set -a; source .env; set +a
fi

# Sanity check
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "✗ OPENAI_API_KEY not set."
  echo "  Either:  export OPENAI_API_KEY=sk-..."
  echo "  Or:      echo 'OPENAI_API_KEY=sk-...' > .env"
  exit 1
fi

# venv setup (first run only)
if [ ! -d venv ]; then
  echo "▶ Creating venv..."
  python3 -m venv venv
  ./venv/bin/pip install --quiet --upgrade pip
  ./venv/bin/pip install --quiet flask openai
fi

# Run
echo "▶ Starting Rover Navigator at http://localhost:5050"
echo "  Stop with Ctrl+C"
echo
./venv/bin/python3 app.py
