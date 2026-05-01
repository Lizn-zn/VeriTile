#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

AXIOM_WHITELIST="${SCRIPT_DIR}/artifact-axiom-whitelist.txt"
THEOREM_LIST="${SCRIPT_DIR}/artifact-theorems.txt"
EXAMPLE_LIST="${SCRIPT_DIR}/artifact-examples.tsv"
DOC_TERMS="${SCRIPT_DIR}/artifact-doc-terms.tsv"

failures=0

ok() {
  printf '[ok] %s\n' "$1"
}

fail() {
  printf '[fail] %s\n' "$1" >&2
  failures=$((failures + 1))
}

is_comment_or_blank() {
  local line="$1"
  [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]]
}

theorem_exists() {
  local file="$1"
  local name="$2"
  [[ -f "${file}" ]] || return 1
  grep -Eq "^[[:space:]]*(private[[:space:]]+)?theorem[[:space:]]+${name}([[:space:]:{(]|$)" "${file}"
}

run_build_no_sorry() {
  local out
  out="$(mktemp)"
  if lake build >"${out}" 2>&1; then
    if grep -qiE "declaration uses 'sorry'|uses sorry|sorryAx" "${out}"; then
      fail "lake build passed but emitted sorry warnings"
      tail -n 40 "${out}" >&2
    else
      ok "lake build passes with no sorry warnings"
    fi
  else
    fail "lake build failed"
    tail -n 80 "${out}" >&2
  fi
  rm -f "${out}"
}

check_axioms() {
  local actual expected unexpected missing
  actual="$(mktemp)"
  expected="$(mktemp)"
  unexpected="$(mktemp)"
  missing="$(mktemp)"

  if [[ -f "${AXIOM_WHITELIST}" ]]; then
    grep -vE '^[[:space:]]*(#|$)' "${AXIOM_WHITELIST}" | sort >"${expected}"
  else
    : >"${expected}"
  fi

  while IFS=: read -r file _line text; do
    # Match declaration axioms only. Mentions in comments/docstrings do not count.
    if [[ "${text}" =~ ^[[:space:]]*axiom[[:space:]]+([^[:space:]:]+) ]]; then
      printf '%s:%s\n' "${file}" "${BASH_REMATCH[1]}"
    fi
  done < <(grep -RInE "^[[:space:]]*axiom[[:space:]]+" VeriTile --include='*.lean' || true) | sort >"${actual}"

  comm -23 "${actual}" "${expected}" >"${unexpected}"
  comm -13 "${actual}" "${expected}" >"${missing}"

  if [[ -s "${unexpected}" ]]; then
    fail "unexpected axiom declarations"
    sed 's/^/  /' "${unexpected}" >&2
  else
    ok "axioms match whitelist"
  fi

  if [[ -s "${missing}" ]]; then
    fail "axiom whitelist contains entries not found in source"
    sed 's/^/  /' "${missing}" >&2
  fi

  rm -f "${actual}" "${expected}" "${unexpected}" "${missing}"
}

check_key_theorems() {
  local line file name
  local before="${failures}"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    is_comment_or_blank "${line}" && continue
    file="${line%%:*}"
    name="${line#*:}"
    if theorem_exists "${file}" "${name}"; then
      :
    else
      fail "missing key theorem ${name} in ${file}"
    fi
  done < "${THEOREM_LIST}"

  if [[ "${failures}" -eq "${before}" ]]; then
    ok "key theorem surface exists"
  fi
}

check_examples_manifest() {
  local line file theorem label
  local before="${failures}"
  while IFS=$'\t' read -r file theorem label || [[ -n "${file:-}" ]]; do
    is_comment_or_blank "${file:-}" && continue
    if [[ ! -f "${file}" ]]; then
      fail "example file missing: ${file}"
      continue
    fi
    if ! theorem_exists "${file}" "${theorem}"; then
      fail "example '${label}' lacks representative theorem ${theorem} in ${file}"
    fi
  done < "${EXAMPLE_LIST}"
  if [[ "${failures}" -eq "${before}" ]]; then
    ok "examples manifest is internally consistent"
  fi
}

check_readme_example_links() {
  local readme link rel missing=0
  for readme in README.md README_zh.md; do
    [[ -f "${readme}" ]] || continue
    while IFS= read -r link; do
      rel="${link#./}"
      if [[ ! -f "${rel}" ]]; then
        fail "${readme} links missing example file: ${rel}"
        missing=1
      fi
    done < <(grep -oE '\./VeriTile/Examples/[^)]*\.lean' "${readme}" | sort -u || true)
  done
  if [[ "${missing}" -eq 0 ]]; then
    ok "README example links resolve"
  fi
}

check_documentation_terms() {
  local line file term
  local before="${failures}"
  while IFS=$'\t' read -r file term || [[ -n "${file:-}" ]]; do
    is_comment_or_blank "${file:-}" && continue
    if [[ ! -f "${file}" ]]; then
      fail "documentation term check missing file: ${file}"
      continue
    fi
    if ! grep -Fq "${term}" "${file}"; then
      fail "${file} is missing required documentation term: ${term}"
    fi
  done < "${DOC_TERMS}"
  if [[ "${failures}" -eq "${before}" ]]; then
    ok "Triton subset/gap docs mention required artifact terms"
  fi
}

run_build_no_sorry
check_axioms
check_key_theorems
check_examples_manifest
check_readme_example_links
check_documentation_terms

if [[ "${failures}" -eq 0 ]]; then
  printf '[ok] artifact checks passed\n'
  exit 0
else
  printf '[fail] artifact checks failed: %d issue(s)\n' "${failures}" >&2
  exit 1
fi
