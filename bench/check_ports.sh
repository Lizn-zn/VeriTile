#!/usr/bin/env bash
#
# Compile and comparator-check every benchmark source, entry and proof file. Independent of
# the main library build: failures here don't
# break `lake build VeriTileFull` or `scripts/check-artifact.sh`, but they
# show up clearly per-kernel.
#
# Each file is replayed independently, including imported dependencies.
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
#   0 — all attempted ports compile and pass the official comparator
#   1 — at least one port failed; the failing kernel names are printed
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

PORTS_ROOT="bench/tritonbench_g"

find_port() {
  python3 bench/port_layout.py proofs "$1"
}

select_ports() {
  if [ "$#" -eq 0 ]; then
    for d in "${PORTS_ROOT}"/*/; do
      find_port "${d}" || return $?
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
  # One elaboration per core, capped by available memory at ~8 GB per comparator worker
  # process (mathlib oleans resident), and capped at 32 — past that the
  # returns are disk-bound. Floor of 1.
  local cores mem_gb by_mem jobs
  cores=$(nproc 2>/dev/null || echo 1)
  mem_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || echo 4)
  by_mem=$((mem_gb / 8))
  jobs=$((cores < by_mem ? cores : by_mem))
  [ "${jobs}" -gt 32 ] && jobs=32
  [ "${jobs}" -lt 1 ] && jobs=1
  printf '%d\n' "${jobs}"
}

JOBS="${CHECK_PORTS_JOBS:-$(default_jobs)}"
if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'CHECK_PORTS_JOBS must be a positive integer: %s\n' "${JOBS}" >&2
  exit 2
fi

TMPDIR_CHECK="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_CHECK}"' EXIT
export PROJECT_ROOT TMPDIR_CHECK

# Materialize the list first so a select_ports failure (unknown kernel name,
# exit 2) propagates instead of vanishing in a pipeline subshell.
port_list="$(select_ports "$@")" || exit "$?"
if [ -z "${port_list}" ]; then
  printf 'No benchmark ports selected\n' >&2
  exit 2
fi

# Each worker prints exactly one status line; single short printf writes are
# atomic within PIPE_BUF, so interleaved output stays line-accurate.
launcher_status=0
results="$(printf '%s\n' "${port_list}" | xargs -P "${JOBS}" -I{} bash -c '
  port="{}"
  kernel="${port#bench/tritonbench_g/}"
  log="${TMPDIR_CHECK}/${kernel//\//_}.log"
  if python3 "${PROJECT_ROOT}/scripts/check_comparator.py" --file "${port}" \
      --workspace "${TMPDIR_CHECK}/judge" >"${log}" 2>&1; then
    cat "${log}" >&2
    printf "  ok    %s\n" "${kernel}"
  else
    cat "${log}" >&2
    printf "  FAIL  %s\n" "${kernel}"
  fi
')" || launcher_status=$?

if [ "${launcher_status}" -ne 0 ]; then
  printf 'Port worker launcher failed (exit %s)\n' "${launcher_status}" >&2
  exit 1
fi

expected="$(printf '%s\n' "${port_list}" | sed 's#^bench/tritonbench_g/##')"
if ! python3 "${SCRIPT_DIR}/check_worker_results.py" "${expected}" "${results}"; then
  exit 1
fi

printf '%s\n' "${results}"

passed=$(printf '%s\n' "${results}" | grep -c '^  ok    ' || true)
failed=$(printf '%s\n' "${results}" | grep -c '^  FAIL  ' || true)

printf '\nTritonBench-G ports: %d ok, %d fail\n' "${passed}" "${failed}"

if [ "${failed}" -gt 0 ]; then
  printf 'Failed kernels:\n'
  printf '%s\n' "${results}" | awk '/^  FAIL  / {print "  - " $2}'
  exit 1
fi
