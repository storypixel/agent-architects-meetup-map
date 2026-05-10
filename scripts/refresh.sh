#!/usr/bin/env bash
# Weekly data refresh: scrape + cluster + commit + push if anything changed.
# Designed for unattended cron / launchd execution.
#
# Setup once on the host machine:
#   cd /path/to/agent-architects-meetup-map
#   uv sync
#   uv run playwright install chromium
#   uv run python scripts/scrape.py --login   # log into Skool, then close window
#
# After that, this script can run unattended.

set -eu

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# Make sure Homebrew + uv are on PATH for launchd contexts
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }

echo "[$(ts)] refresh starting in $REPO"

uv run python scripts/scrape.py
uv run python scripts/cluster.py

if git diff --quiet members.csv data.json; then
  echo "[$(ts)] no data changes; skipping commit"
  exit 0
fi

git add members.csv data.json
git commit -m "Weekly data refresh: $(date -u +%Y-%m-%d)"
git push origin main

echo "[$(ts)] refresh complete; pushed to origin"
