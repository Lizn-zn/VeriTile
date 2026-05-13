#!/usr/bin/env bash
#
# Audit the currently-ported TritonBench-G files against the repository-level
# completion gates that can be checked mechanically.
#
# This does not prove line-by-line semantic faithfulness. That still requires
# review_criteria.md-driven human review, and the remaining proof obligations
# are tracked in bench/tritonbench_g/proof_blockers.md.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
PORTS_ROOT="${PROJECT_ROOT}/bench/tritonbench_g"

cd "${PROJECT_ROOT}"

failures=0

py_count=$(find "${PORTS_ROOT}" -maxdepth 2 -name '*.py' | wc -l | tr -d ' ')
lean_count=$(find "${PORTS_ROOT}" -maxdepth 2 -name '*.lean' | wc -l | tr -d ' ')

if [ "${py_count}" != "${lean_count}" ]; then
  printf 'FAIL python/lean port count mismatch: %s py, %s lean\n' "${py_count}" "${lean_count}"
  failures=$((failures + 1))
else
  printf 'ok python/lean port count: %s\n' "${py_count}"
fi

if bench/check_ports.sh; then
  printf 'ok port elaboration\n'
else
  printf 'FAIL port elaboration\n'
  failures=$((failures + 1))
fi

if rg -n 'True := by|trivial|sorry|admit' bench/tritonbench_g -g '*.lean'; then
  printf 'FAIL placeholder proof scan found matches\n'
  failures=$((failures + 1))
else
  printf 'ok placeholder proof scan\n'
fi

missing_surface=()
while IFS= read -r lean_file; do
  if ! rg -q 'ComputeCorrect\.Realizes|ComputeRefine\.Realizes|ComputeCorrect\.General|correct_target' "${lean_file}"; then
    missing_surface+=("${lean_file}")
  fi
done < <(find "${PORTS_ROOT}" -maxdepth 2 -name '*.lean' | sort)

if [ "${#missing_surface[@]}" -gt 0 ]; then
  printf 'FAIL missing correctness surface:\n'
  printf '  %s\n' "${missing_surface[@]}"
  failures=$((failures + 1))
else
  printf 'ok correctness surface scan\n'
fi

if [ "${failures}" -gt 0 ]; then
  exit 1
fi

printf 'TritonBench-G audit gates passed\n'
