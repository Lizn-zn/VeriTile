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

stale_readmes=()
while IFS= read -r readme; do
  dir="${readme%/README.md}"
  if find "${dir}" -maxdepth 1 -name '*.lean' | rg -q . &&
      rg -q 'Status: TODO' "${readme}"; then
    stale_readmes+=("${readme}")
  fi
done < <(find "${PORTS_ROOT}" -mindepth 2 -maxdepth 2 -name 'README.md' | sort)

while IFS= read -r readme; do
  dir="${readme%/README_zh.md}"
  if find "${dir}" -maxdepth 1 -name '*.lean' | rg -q . &&
      rg -q '状态:TODO' "${readme}"; then
    stale_readmes+=("${readme}")
  fi
done < <(find "${PORTS_ROOT}" -mindepth 2 -maxdepth 2 -name 'README_zh.md' | sort)

if [ "${#stale_readmes[@]}" -gt 0 ]; then
  printf 'FAIL compiled ports with TODO README status:\n'
  printf '  %s\n' "${stale_readmes[@]}"
  failures=$((failures + 1))
else
  printf 'ok compiled-port README status scan\n'
fi

undocumented_cast_gaps=()
while IFS= read -r py_file; do
  if ! rg -q '\.to\(tl\.float32\)' "${py_file}"; then
    continue
  fi
  dir="${py_file%/*}"
  lean_file="$(find "${dir}" -maxdepth 1 -name '*.lean' | head -n 1)"
  if [ -z "${lean_file}" ]; then
    continue
  fi
  if rg -q '\.to\(tl\.float32\)|to\(tl\.float32\)|\.to\(tl.float32\)|to\(tl.float32\)' "${lean_file}"; then
    continue
  fi
  if rg -q 'outside this|quant_policy = 0|unquantized path|quant_policy.*4/8' "${lean_file}"; then
    continue
  fi
  undocumented_cast_gaps+=("${py_file} -> ${lean_file}")
done < <(find "${PORTS_ROOT}" -mindepth 2 -maxdepth 2 -name '*.py' | sort)

if [ "${#undocumented_cast_gaps[@]}" -gt 0 ]; then
  printf 'FAIL .to(tl.float32) missing from Lean without documented slice/scope:\n'
  printf '  %s\n' "${undocumented_cast_gaps[@]}"
  failures=$((failures + 1))
else
  printf 'ok documented float32 cast coverage scan\n'
fi

if rg -n 'tl\.load\([^\n]*dtype\s*=' bench/tritonbench_g -g '*.lean'; then
  printf 'FAIL Lean tl.load dtype annotations found; review_criteria.md only permits them when present upstream\n'
  failures=$((failures + 1))
else
  printf 'ok no Lean-only tl.load dtype annotations\n'
fi

if rg -n 'keep_dims\s*=\s*true|keepDims|keep_dims' bench/tritonbench_g -g '*.lean'; then
  printf 'FAIL keep_dims-style reduction found; review_criteria.md requires source-shape-preserving syntax\n'
  failures=$((failures + 1))
else
  printf 'ok no keep_dims reduction substitutions\n'
fi

undocumented_iadd_gaps=()
while IFS= read -r py_file; do
  if ! rg -q '\+=' "${py_file}"; then
    continue
  fi
  dir="${py_file%/*}"
  lean_file="$(find "${dir}" -maxdepth 1 -name '*.lean' | head -n 1)"
  if [ -z "${lean_file}" ]; then
    continue
  fi
  if rg -q '\+=' "${lean_file}"; then
    continue
  fi
  if rg -q 'slice|outside this|branch|precomputed|Surface transcription' "${lean_file}"; then
    continue
  fi
  undocumented_iadd_gaps+=("${py_file} -> ${lean_file}")
done < <(find "${PORTS_ROOT}" -mindepth 2 -maxdepth 2 -name '*.py' | sort)

if [ "${#undocumented_iadd_gaps[@]}" -gt 0 ]; then
  printf 'FAIL Python += missing from Lean without documented slice/scope:\n'
  printf '  %s\n' "${undocumented_iadd_gaps[@]}"
  failures=$((failures + 1))
else
  printf 'ok documented += coverage scan\n'
fi

if [ "${failures}" -gt 0 ]; then
  exit 1
fi

printf 'TritonBench-G audit gates passed\n'
