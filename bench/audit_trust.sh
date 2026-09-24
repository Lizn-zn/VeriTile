#!/usr/bin/env bash
#
# External machine-checkable TRUST AUDIT for the standalone bench corpus.
#
# The library theorems are audited in-tree by VeriTile/Meta/TrustReport.lean
# (`lake build VeriTile.Meta.TrustReport`). This script covers the OTHER
# population — the standalone, single-file bench elaborations that are NOT
# importable:
#
#   * bench/tritonbench_g/*/*.lean   (the 173 TritonBench-G ports)
#   * bench/examples/*.lean          (the kernel showcases)
#   * bench/tests/*.lean             (infra smoke tests / regression gates)
#
# For each file the shared comparator driver emits a snapshot copy that adds
# `import VeriTile.Meta.StatementAudit` and appends `#auditModuleAxioms` and
# `#auditModuleSpecs` for environment-based discovery, compiles it, and requires
# official comparator export/replay. The port files themselves are never modified —
# the corpus stays clean.
#
# A `#axiomsClean` failure means a `sorry`/smuggled axiom leaked into a proof
# the corpus calls complete: a REAL soundness finding. Do not paper it over.
#
# Independent single-file elaborations run in parallel (see bench/check_ports.sh
# for the same concurrency model). Override with AUDIT_TRUST_JOBS=N.
#
# Usage:
#   bench/audit_trust.sh                 # audit the whole bench corpus
#   bench/audit_trust.sh <kernel> ...    # audit only named tritonbench_g kernels
#   AUDIT_TRUST_JOBS=4 bench/audit_trust.sh
#   AUDIT_TRUST_SHARD_COUNT=4 AUDIT_TRUST_SHARD_INDEX=0 bench/audit_trust.sh
# Shards use zero-based indices; every shard must pass to audit the full corpus.
#
# Exit codes:
#   0 — every audited file passes its trust gates and official comparator replay
#   1 — at least one file failed; failing names are printed
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

PORTS_ROOT="bench/tritonbench_g"
EXAMPLES_ROOT="bench/examples"
TESTS_ROOT="bench/tests"

select_targets() {
  if [ "$#" -eq 0 ]; then
    find "${PORTS_ROOT}" -mindepth 2 -maxdepth 2 -type f -name '*.lean' | sort
    find "${EXAMPLES_ROOT}" -maxdepth 1 -type f -name '*.lean' | sort
    find "${TESTS_ROOT}" -maxdepth 1 -type f -name '*.lean' | sort
  else
    for name in "$@"; do
      local lean
      lean=$(find "${PORTS_ROOT}/${name}/" -maxdepth 1 -type f -name '*.lean' 2>/dev/null | head -n 1)
      if [ -n "${lean}" ]; then
        printf '%s\n' "${lean}"
      else
        printf 'no .lean file for bench target: %s\n' "${name}" >&2
        exit 2
      fi
    done
  fi
}

default_jobs() {
  local cores mem_gb by_mem jobs
  cores=$(nproc 2>/dev/null || echo 1)
  mem_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || echo 4)
  by_mem=$((mem_gb / 8))
  jobs=$((cores < by_mem ? cores : by_mem))
  [ "${jobs}" -gt 32 ] && jobs=32
  [ "${jobs}" -lt 1 ] && jobs=1
  printf '%d\n' "${jobs}"
}

JOBS="${AUDIT_TRUST_JOBS:-$(default_jobs)}"
if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'AUDIT_TRUST_JOBS must be a positive integer: %s\n' "${JOBS}" >&2
  exit 2
fi

target_list="$(select_targets "$@")" || exit "$?"
if [ -z "${target_list}" ]; then
  printf 'No trust audit targets selected\n' >&2
  exit 2
fi

audit_scope='bench corpus'
if [[ -v AUDIT_TRUST_SHARD_COUNT || -v AUDIT_TRUST_SHARD_INDEX ]]; then
  if [ "$#" -ne 0 ]; then
    printf 'Trust audit sharding cannot be combined with named targets\n' >&2
    exit 2
  fi
  target_list="$(printf '%s\n' "${target_list}" | python3 "${SCRIPT_DIR}/select_audit_shard.py" \
    "${AUDIT_TRUST_SHARD_COUNT-}" "${AUDIT_TRUST_SHARD_INDEX-}")" || exit "$?"
  audit_scope="bench corpus shard ${AUDIT_TRUST_SHARD_INDEX}/${AUDIT_TRUST_SHARD_COUNT} (zero-based)"
  printf 'Selected %s: %s files\n' "${audit_scope}" "$(printf '%s\n' "${target_list}" | wc -l)"
fi

TMPDIR_AUDIT="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_AUDIT}"' EXIT

export PROJECT_ROOT TMPDIR_AUDIT

launcher_status=0
results="$(printf '%s\n' "${target_list}" | xargs -P "${JOBS}" -I{} bash -c '
  src="{}"
  rel="${src#'"${PROJECT_ROOT}"'/}"
  # a unique temp name: <parent-dir>__<basename>
  tag="$(basename "$(dirname "${src}")")__$(basename "${src}")"
  tmp="${TMPDIR_AUDIT}/${tag}"
  if python3 "${PROJECT_ROOT}/scripts/check_comparator.py" --file "${src}" --trust \
      --workspace "${TMPDIR_AUDIT}/judge" > "${tmp}.log" 2>&1; then
    summary=$(sed -n "/^Axiom audit:/p; /^Spec audit:/p; /^Comparator:/p" "${tmp}.log" | paste -sd ";")
    printf "%s: %s\n" "${rel}" "${summary}" >&2
    printf "  ok    %s\n" "${rel}"
  else
    cat "${tmp}.log" >&2
    printf "  FAIL  %s\n" "${rel}"
  fi
')" || launcher_status=$?

if [ "${launcher_status}" -ne 0 ]; then
  printf 'Trust worker launcher failed (exit %s)\n' "${launcher_status}" >&2
  exit 1
fi

expected="$(printf '%s\n' "${target_list}" | sed "s|^${PROJECT_ROOT}/||")"
if ! python3 "${SCRIPT_DIR}/check_worker_results.py" "${expected}" "${results}"; then
  exit 1
fi

printf '%s\n' "${results}" | sort

passed=$(printf '%s\n' "${results}" | grep -c '^  ok    ' || true)
failed=$(printf '%s\n' "${results}" | grep -c '^  FAIL  ' || true)

printf '\nTrust audit (%s): %d ok, %d fail\n' "${audit_scope}" "${passed}" "${failed}"

if [ "${failed}" -gt 0 ]; then
  printf 'Files whose trust gate FAILED (inspect diagnostics for proof or infrastructure failure):\n'
  printf '%s\n' "${results}" | awk '/^  FAIL  / {print "  - " $2}'
  exit 1
fi
