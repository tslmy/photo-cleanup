#!/bin/bash
# Trial run: sets everything up and analyzes the newest 500 photos.
# Only reads the library – no keywords, nothing is deleted.
# Usage: ./trial_run.sh [N] [--library PATH]
set -eo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "❌ uv not found. Install it first: brew install uv"
  exit 1
fi

echo "→ Installing packages (one time, a few minutes) …"
uv sync -q

N="${1:-500}"; shift || true
echo "→ Starting trial run (newest $N photos, report only) …"
uv run photo_cleanup.py scan --limit "$N" --no-albums "$@" 2>&1 | tee trial_run.log
echo
echo "✅ Done. Report is in ~/.photo_cleanup/reports/ – log in trial_run.log"
