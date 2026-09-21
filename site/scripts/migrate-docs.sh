#!/usr/bin/env bash
# Sync English repository design notes into the site. Pass --check in CI.
set -euo pipefail
exec python3 "$(dirname "$0")/sync-docs.py" "$@"
