#!/usr/bin/env bash
# One-touch dev server for the VeriTile docs site.
#
#   ./site/scripts/dev.sh           # install (if needed) + start dev server
#   ./site/scripts/dev.sh build     # one-shot production build
#   ./site/scripts/dev.sh preview   # build + serve the production output
#
# Requires Bun (https://bun.sh). Node/npm are not used.

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v bun >/dev/null 2>&1; then
  echo "error: bun is not installed. Install from https://bun.sh, then re-run." >&2
  exit 1
fi

if [ ! -d node_modules ]; then
  echo "→ installing site dependencies (one-time)"
  bun install --frozen-lockfile
fi

mode="${1:-dev}"
case "$mode" in
  dev)     bun run dev ;;
  build)   bun run build ;;
  preview) bun run build && bun run preview ;;
  *)
    echo "usage: $0 [dev|build|preview]" >&2
    exit 64
    ;;
esac
