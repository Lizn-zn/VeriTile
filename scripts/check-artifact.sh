#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

AXIOM_WHITELIST="${SCRIPT_DIR}/artifact-axiom-whitelist.txt"
KERNEL_MANIFEST="${SCRIPT_DIR}/kernel-manifest.tsv"
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
  if lake build VeriTile VeriTileFull >"${out}" 2>&1; then
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

valid_manifest_kind() {
  case "$1" in
    correct|refine|math|launch|trace|safety|frame) return 0 ;;
    *) return 1 ;;
  esac
}

valid_manifest_status() {
  case "$1" in
    proven|projected|test-gap|blocked|smoke) return 0 ;;
    *) return 1 ;;
  esac
}

check_kernel_manifest() {
  local id file theorem kind status source source_ref config label notes extra
  local before="${failures}"
  local ids
  ids="$(mktemp)"

  if [[ ! -f "${KERNEL_MANIFEST}" ]]; then
    fail "kernel manifest missing: ${KERNEL_MANIFEST}"
    rm -f "${ids}"
    return
  fi

  while IFS=$'\t' read -r id file theorem kind status source source_ref config label notes extra ||
      [[ -n "${id:-}" ]]; do
    is_comment_or_blank "${id:-}" && continue
    if [[ "${id}" == "id" ]]; then
      continue
    fi

    if [[ -n "${extra:-}" ]]; then
      fail "kernel manifest row ${id} has too many columns"
      continue
    fi
    if [[ -z "${id}" || -z "${file}" || -z "${theorem}" || -z "${kind}" ||
          -z "${status}" || -z "${source}" || -z "${source_ref}" ||
          -z "${config}" || -z "${label}" || -z "${notes}" ]]; then
      fail "kernel manifest row has empty required field: ${id:-<missing id>}"
      continue
    fi
    if grep -Fxq "${id}" "${ids}"; then
      fail "duplicate kernel manifest id: ${id}"
    else
      printf '%s\n' "${id}" >>"${ids}"
    fi
    if ! valid_manifest_kind "${kind}"; then
      fail "kernel manifest ${id} has invalid kind: ${kind}"
    fi
    if ! valid_manifest_status "${status}"; then
      fail "kernel manifest ${id} has invalid status: ${status}"
    fi
    if [[ ! -f "${file}" ]]; then
      fail "kernel manifest ${id} missing file: ${file}"
      continue
    fi
    if ! theorem_exists "${file}" "${theorem}"; then
      fail "kernel manifest ${id} missing theorem ${theorem} in ${file}"
    fi
  done < "${KERNEL_MANIFEST}"

  rm -f "${ids}"

  if [[ "${failures}" -eq "${before}" ]]; then
    ok "kernel manifest is internally consistent"
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
check_kernel_manifest
check_readme_example_links
check_documentation_terms

if [[ "${failures}" -eq 0 ]]; then
  printf '[ok] artifact checks passed\n'
  exit 0
else
  printf '[fail] artifact checks failed: %d issue(s)\n' "${failures}" >&2
  exit 1
fi
