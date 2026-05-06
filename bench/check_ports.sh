#!/usr/bin/env bash
#
# Compile every benchmark port `Port.lean` against the parent project's
# `lake env`. Independent of the main library build: failures here don't
# break `lake build VeriTileFull` or `scripts/check-artifact.sh`, but they
# show up clearly per-kernel.
#
# Usage:
#   bench/check_ports.sh                  # compile all ports
#   bench/check_ports.sh <kernel> ...     # compile only the named kernels
#
# Exit codes:
#   0 — all attempted ports compile
#   1 — at least one port failed; the failing kernel names are printed
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

PORTS_ROOT="bench/tritonbench_g"

find_port() {
  # The Lean port lives in `<dir>/<CamelCase>.lean`. We don't hard-code the
  # CamelCase translation here — just take the first .lean file in the dir
  # (each port dir is expected to have at most one).
  local dir="$1"
  local lean
  lean=$(find "${dir}" -maxdepth 1 -type f -name '*.lean' | head -n 1)
  [ -n "${lean}" ] && printf '%s\n' "${lean}"
}

select_ports() {
  if [ "$#" -eq 0 ]; then
    for d in "${PORTS_ROOT}"/*/; do
      find_port "${d}"
    done
  else
    for name in "$@"; do
      local lean
      lean=$(find_port "${PORTS_ROOT}/${name}/")
      if [ -n "${lean}" ]; then
        printf '%s\n' "${lean}"
      else
        printf 'no .lean port for kernel: %s\n' "${name}" >&2
        exit 2
      fi
    done
  fi
}

passed=0
failed=0
fail_list=()

while IFS= read -r port; do
  kernel="$(basename "$(dirname "${port}")")"
  if lake env lean "${port}" >/dev/null 2>&1; then
    printf '  ok    %s\n' "${kernel}"
    passed=$((passed + 1))
  else
    printf '  FAIL  %s\n' "${kernel}"
    failed=$((failed + 1))
    fail_list+=("${kernel}")
  fi
done < <(select_ports "$@")

printf '\nTritonBench-G ports: %d ok, %d fail\n' "${passed}" "${failed}"

if [ "${failed}" -gt 0 ]; then
  printf 'Failed kernels:\n'
  for k in "${fail_list[@]}"; do
    printf '  - %s\n' "${k}"
  done
  exit 1
fi
