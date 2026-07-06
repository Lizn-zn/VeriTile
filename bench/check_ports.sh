#!/usr/bin/env bash
#
# Compile every benchmark port `Port.lean` against the parent project's
# `lake env`. Independent of the main library build: failures here don't
# break `lake build VeriTileFull` or `scripts/check-artifact.sh`, but they
# show up clearly per-kernel.
#
# Ports are independent single-file elaborations, so they run in parallel.
# Concurrency defaults to a memory-capped heuristic (each `lake env lean`
# on a mathlib-heavy port peaks at a few GB); override with
# `CHECK_PORTS_JOBS=N`.
#
# Usage:
#   bench/check_ports.sh                  # compile all ports
#   bench/check_ports.sh <kernel> ...     # compile only the named kernels
#   CHECK_PORTS_JOBS=4 bench/check_ports.sh
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

default_jobs() {
  # One elaboration per core, capped by available memory at ~4 GB per lean
  # process (mathlib oleans resident), and capped at 32 — past that the
  # returns are disk-bound. Floor of 1.
  local cores mem_gb by_mem jobs
  cores=$(nproc 2>/dev/null || echo 1)
  mem_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || echo 4)
  by_mem=$((mem_gb / 4))
  jobs=$((cores < by_mem ? cores : by_mem))
  [ "${jobs}" -gt 32 ] && jobs=32
  [ "${jobs}" -lt 1 ] && jobs=1
  printf '%d\n' "${jobs}"
}

JOBS="${CHECK_PORTS_JOBS:-$(default_jobs)}"

# Materialize the list first so a select_ports failure (unknown kernel name,
# exit 2) propagates instead of vanishing in a pipeline subshell.
port_list="$(select_ports "$@")" || exit "$?"

# Each worker prints exactly one status line; single short printf writes are
# atomic within PIPE_BUF, so interleaved output stays line-accurate.
results="$(printf '%s\n' "${port_list}" | xargs -P "${JOBS}" -I{} bash -c '
  port="{}"
  kernel="$(basename "$(dirname "${port}")")"
  if lake env lean "${port}" >/dev/null 2>&1; then
    printf "  ok    %s\n" "${kernel}"
  else
    printf "  FAIL  %s\n" "${kernel}"
  fi
')"

printf '%s\n' "${results}"

passed=$(printf '%s\n' "${results}" | grep -c '^  ok    ' || true)
failed=$(printf '%s\n' "${results}" | grep -c '^  FAIL  ' || true)

printf '\nTritonBench-G ports: %d ok, %d fail\n' "${passed}" "${failed}"

if [ "${failed}" -gt 0 ]; then
  printf 'Failed kernels:\n'
  printf '%s\n' "${results}" | awk '/^  FAIL  / {print "  - " $2}'
  exit 1
fi
