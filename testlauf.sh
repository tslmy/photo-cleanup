#!/bin/bash
# Testlauf: richtet alles ein und analysiert die neuesten 500 Fotos.
# Liest die Mediathek nur – keine Schlagwörter, nichts wird gelöscht.
# Usage: ./testlauf.sh [N] [--library PATH]
set -eo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "❌ uv nicht gefunden. Bitte zuerst: brew install uv"
  exit 1
fi

echo "→ Installiere Pakete (einmalig, ein paar Minuten) …"
uv sync -q

N="${1:-500}"; shift || true
echo "→ Starte Testlauf (neueste $N Fotos, nur Bericht) …"
uv run photo_cleanup.py scan --limit "$N" --no-albums "$@" 2>&1 | tee testlauf.log
echo
echo "✅ Fertig. Bericht liegt in ~/.photo_cleanup/reports/ – Log in testlauf.log"
