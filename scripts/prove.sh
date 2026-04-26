#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/prove.sh <lean_file> [--max-cycles N] [--prompt "extra"]
#
# Wraps `claude -p "/lean4:autoprove ..."` to attempt closing sorries in a
# Lean file. Captures the JSON output to Logs/, parses success/fail.
# Exits 0 on success, 1 on fail. Designed for benchmark eval (see PLAN.md
# §LLM benchmark protocol).

usage() {
  cat >&2 <<'EOF'
Usage: scripts/prove.sh <lean_file> [--max-cycles N] [--prompt "extra"]
  Wraps /lean4:autoprove to close sorries in a Lean file.
  --max-cycles N    Stop after N proof cycles (default 5)
  --prompt "..."    Extra prompt text appended to the autoprove invocation
EOF
  exit 1
}

[ "$#" -ge 1 ] || usage

LEAN_FILE=""
MAX_CYCLES="5"
EXTRA_PROMPT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --max-cycles)
      [ "$#" -ge 2 ] || { echo "Error: --max-cycles needs a value" >&2; usage; }
      MAX_CYCLES="$2"; shift 2 ;;
    --prompt)
      [ "$#" -ge 2 ] || { echo "Error: --prompt needs a value" >&2; usage; }
      EXTRA_PROMPT="$2"; shift 2 ;;
    -h|--help) usage ;;
    --*) echo "Error: unknown flag '$1'" >&2; usage ;;
    *)
      if [ -z "${LEAN_FILE}" ]; then
        LEAN_FILE="$1"; shift
      else
        echo "Error: unexpected positional '$1'" >&2; usage
      fi ;;
  esac
done

[ -n "${LEAN_FILE}" ] || { echo "Error: missing <lean_file>" >&2; usage; }
[ -e "${LEAN_FILE}" ] || { echo "Error: file not found: ${LEAN_FILE}" >&2; exit 1; }

LEAN_FILE_ABS="$(cd "$(dirname "${LEAN_FILE}")" && pwd)/$(basename "${LEAN_FILE}")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
LOGS_DIR="${PROJECT_ROOT}/Logs"
mkdir -p "${LOGS_DIR}"

BASE="$(basename "${LEAN_FILE_ABS}" .lean)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_JSON="${LOGS_DIR}/${BASE}_${TIMESTAMP}.json"

CMD="/lean4:autoprove ${LEAN_FILE_ABS} --max-cycles=${MAX_CYCLES} --commit=never --planning=off --review-source=none"
if [ -n "${EXTRA_PROMPT}" ]; then
  CMD="${CMD} ${EXTRA_PROMPT}"
fi

START=$(date +%s)
set +e
claude -p "${CMD}" \
  --dangerously-skip-permissions \
  --max-budget-usd 10.00 \
  --output-format stream-json \
  --include-partial-messages \
  --verbose \
  > "${OUT_JSON}"
EXIT=$?
set -e
END=$(date +%s)
DURATION=$((END - START))

# Parse the result line for success/fail. The /lean4:autoprove command emits
# a final result message with structured info; we look for the last
# {"type":"result", ...} line. Keep parsing simple for v0.1.
RESULT_LINE=$(tail -n 50 "${OUT_JSON}" | grep -m1 '"type"[: ]*"result"' || true)
if [ -n "${RESULT_LINE}" ]; then
  SUBTYPE=$(printf '%s' "${RESULT_LINE}" | jq -r '.subtype // "unknown"' 2>/dev/null || echo "unknown")
  RESULT_TEXT=$(printf '%s' "${RESULT_LINE}" | jq -r '.result // ""' 2>/dev/null || echo "")
else
  SUBTYPE="no-result-line"
  RESULT_TEXT=""
fi

# Heuristic success detection: clean exit AND result subtype isn't error AND
# the result text doesn't say "fail/stuck/unable to close/sorry remains".
if [ "${EXIT}" -eq 0 ] && [ "${SUBTYPE}" != "error" ] && \
   ! printf '%s' "${RESULT_TEXT}" | grep -qiE 'fail|stuck|unable to close|sorry remains'; then
  echo "[SUCCESS] ${LEAN_FILE} (${DURATION}s, log: ${OUT_JSON})"
  exit 0
else
  echo "[FAIL] ${LEAN_FILE} (exit=${EXIT}, subtype=${SUBTYPE}, ${DURATION}s, log: ${OUT_JSON})"
  exit 1
fi
