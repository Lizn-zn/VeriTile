#!/usr/bin/env bash
# One-shot: rebuild the Verso deck and reload the browser tab.
#
#   ./verso/rebuild.sh           # full lake build + regen + browser reload
#   ./verso/rebuild.sh --css     # CSS-only fast path (no lake build)
#
# Assumes a static server is already serving verso/_site on 127.0.0.1:8765
# (e.g. `python3 -m http.server 8765 --bind 127.0.0.1`).

set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-}" == "--css" ]]; then
  echo "→ syncing static_files/ → _site/static/"
  cp -R static_files/* _site/static/
else
  echo "→ lake build"
  lake build

  echo "→ regenerating _site/"
  ./.lake/build/bin/veritile-overview --output _site
fi

# Best-effort browser refresh — find any Chrome/Safari tab pointing at our
# server and reload it. Silently skipped if AppleScript permissions denied.
osascript <<'OSA' 2>/dev/null || true
tell application "System Events"
  set chromeRunning to (exists process "Google Chrome")
  set safariRunning to (exists process "Safari")
end tell

if chromeRunning then
  tell application "Google Chrome"
    repeat with w in windows
      repeat with t in tabs of w
        try
          set u to URL of t
          if u contains "127.0.0.1:8765" or u contains "localhost:8765" then
            tell t to reload
          end if
        end try
      end repeat
    end repeat
  end tell
end if

if safariRunning then
  tell application "Safari"
    repeat with w in windows
      repeat with t in tabs of w
        try
          set u to URL of t
          if u contains "127.0.0.1:8765" or u contains "localhost:8765" then
            set URL of t to u
          end if
        end try
      end repeat
    end repeat
  end tell
end if
OSA

echo "✓ done"
