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

# Hard dependency check: a missing rg/python3 makes `if rg ...` conditions
# read as "no match", silently turning scans into false passes.
for dep in rg python3; do
  if ! command -v "${dep}" >/dev/null 2>&1; then
    printf 'FAIL required tool not installed: %s\n' "${dep}"
    exit 1
  fi
done

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

# Placeholder-proof scan. Operates on comment-stripped Lean source so prose
# like "sorry-free" in docstrings cannot mask or fake a hit, and matches
# `trivial` only where it is the entire proof (`:= trivial`, `:= by trivial`),
# not where it appears as a proof term inside a larger tactic application.
if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])

def strip_lean_comments(text: str) -> str:
    """Blank out `--` line comments and (nested) `/- ... -/` block comments,
    preserving line structure so reported line numbers stay accurate."""
    out = []
    i = 0
    n = len(text)
    depth = 0
    while i < n:
        two = text[i:i + 2]
        if depth == 0 and two == "--":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if two == "/-":
            depth += 1
            out.append("  ")
            i += 2
            continue
        if depth > 0 and two == "-/":
            depth -= 1
            out.append("  ")
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        else:
            out.append(text[i] if text[i] == "\n" else " ")
        i += 1
    return "".join(out)

# `sorry` / `admit` as code tokens; `True := by` placeholder goals; `trivial`
# standing alone as the entire proof (term-mode `:= trivial`, one-line
# `:= by trivial`, or the two-line `:= by` / lone indented `trivial` form).
line_patterns = (
    re.compile(r"\bsorry\b"),
    re.compile(r"\badmit\b"),
    re.compile(r"True\s*:=\s*by\b"),
    re.compile(r":=\s*(?:by\s+)?trivial\s*$"),
)
whole_proof_trivial = re.compile(r":=\s*by[ \t]*\n\s*trivial[ \t]*\n(?![ \t])")

matches = []
for lean_file in sorted(root.glob("*/*.lean")):
    code = strip_lean_comments(lean_file.read_text())
    for lineno, line in enumerate(code.splitlines(), 1):
        for pattern in line_patterns:
            if pattern.search(line):
                matches.append(f"{lean_file}:{lineno}:{line.strip()}")
                break
    for m in whole_proof_trivial.finditer(code):
        lineno = code.count("\n", 0, m.start()) + 1
        matches.append(f"{lean_file}:{lineno}:whole-proof `by trivial`")

if matches:
    for entry in sorted(set(matches)):
        print(entry)
    sys.exit(1)
PY
then
  printf 'ok placeholder proof scan\n'
else
  printf 'FAIL placeholder proof scan found matches\n'
  failures=$((failures + 1))
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

if python3 bench/check_proof_gap_manifest.py; then
  printf 'ok proof-gap manifest scan\n'
else
  printf 'FAIL proof-gap manifest scan\n'
  failures=$((failures + 1))
fi

known_algorithm_blockers=()

unexpected_algorithm_blockers=()
while IFS= read -r lean_file; do
  if ! rg -q '\(hAlg\s*:' "${lean_file}"; then
    continue
  fi
  rel="${lean_file#${PROJECT_ROOT}/}"
  known=false
  if [ "${#known_algorithm_blockers[@]}" -gt 0 ]; then
    for blocker in "${known_algorithm_blockers[@]}"; do
      if [ "${rel}" = "${blocker}" ]; then
        known=true
        break
      fi
    done
  fi
  if [ "${known}" = false ]; then
    unexpected_algorithm_blockers+=("${rel}")
  fi
done < <(find "${PORTS_ROOT}" -maxdepth 2 -name '*.lean' | sort)

if [ "${#unexpected_algorithm_blockers[@]}" -gt 0 ]; then
  printf 'FAIL unexpected algorithm-layer hAlg blockers:\n'
  printf '  %s\n' "${unexpected_algorithm_blockers[@]}"
  failures=$((failures + 1))
else
  printf 'ok known algorithm-layer blocker scan: %s documented\n' \
    "${#known_algorithm_blockers[@]}"
fi


# A port that deliberately deviates from a literal transcription of the
# upstream Python surface (inlined helper JIT, constexpr-path specialization,
# antiquoted in-body constants, ...) must declare that with an explicit
# machine-readable preamble line
#     Translation-surface blocker: <what deviates and why it is faithful>
# before the first `triton {`. The same marker is the ONLY exemption the
# textual py<->lean surface scans below accept, and every marker must be
# registered in proof_blockers.md and completion_audit.md.
if python3 - "${PORTS_ROOT}" "${PORTS_ROOT}/proof_blockers.md" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
doc = Path(sys.argv[2]).read_text()
marker_re = re.compile(r"^\s*(?:--\s*)?Translation-surface blocker:(.*)$", re.M)

problems = []
for lean_file in sorted(root.glob("*/*.lean")):
    text = lean_file.read_text()
    preamble = text[:text.find("triton {")] if "triton {" in text else text
    port_name = lean_file.parent.name
    markers = marker_re.findall(preamble)
    if not markers:
        continue
    if any(not body.strip() for body in markers):
        problems.append(
            f"{port_name}: Translation-surface blocker marker has no description"
        )
    if f"`{port_name}`" not in doc:
        problems.append(
            f"{port_name}: Translation-surface blocker marker lacks proof_blockers.md entry"
        )

if problems:
    for problem in problems:
        print(problem)
    sys.exit(1)
PY
then
  printf 'ok documented translation-surface blocker scan\n'
else
  printf 'FAIL documented translation-surface blocker scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" "${PORTS_ROOT}/completion_audit.md" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
doc = Path(sys.argv[2]).read_text()
marker_re = re.compile(r"^\s*(?:--\s*)?Translation-surface blocker:\s*\S", re.M)

actual = []
for lean_file in sorted(root.glob("*/*.lean")):
    text = lean_file.read_text()
    preamble = text[:text.find("triton {")] if "triton {" in text else text
    if marker_re.search(preamble):
        actual.append(lean_file.parent.name)

listed = []
in_block = False
seen_item = False
for line in doc.splitlines():
    if line.strip() == "The current documented blocker set is:":
        in_block = True
        continue
    if not in_block:
        continue
    if not line.strip() and not seen_item:
        continue
    if not line.strip():
        break
    m = re.match(r"- `([^`]+)`", line.strip())
    if m:
        listed.append(m.group(1))
        seen_item = True

actual_set = set(actual)
listed_set = set(listed)
missing = sorted(actual_set - listed_set)
stale = sorted(listed_set - actual_set)
dupes = sorted({name for name in listed if listed.count(name) > 1})
if missing or stale or dupes:
    for name in missing:
        print(f"{name}: completion_audit.md missing active translation-surface blocker")
    for name in stale:
        print(f"{name}: completion_audit.md lists stale translation-surface blocker")
    for name in dupes:
        print(f"{name}: completion_audit.md lists duplicate translation-surface blocker")
    sys.exit(1)
PY
then
  printf 'ok completion audit blocker list scan\n'
else
  printf 'FAIL completion audit blocker list scan\n'
  failures=$((failures + 1))
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

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import ast
import re
import sys

root = Path(sys.argv[1])

def is_triton_jit(dec: ast.expr) -> bool:
    if isinstance(dec, ast.Call):
        dec = dec.func
    return (
        isinstance(dec, ast.Attribute) and dec.attr == "jit"
    ) or (
        isinstance(dec, ast.Name) and dec.id == "jit"
    )

def python_jit_names(text: str) -> set[str]:
    tree = ast.parse(text)
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(is_triton_jit(dec) for dec in node.decorator_list)
    }

def lean_first_preamble(text: str) -> str:
    idx = text.find("triton {")
    return text[:idx] if idx >= 0 else text

def target_kernel_name(preamble: str):
    matches = re.findall(r"\.py`'s[^`]*`([^`]+)`", preamble, re.S)
    return matches[-1] if matches else None

failures = []
missing_docs = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    target = target_kernel_name(lean_first_preamble(lean_file.read_text()))
    if target is None:
        missing_docs.append((py_file, lean_file))
        continue
    jit_names = python_jit_names(py_file.read_text())
    if target not in jit_names:
        failures.append((py_file, lean_file, target, sorted(jit_names)))

if missing_docs or failures:
    for py_file, lean_file in missing_docs:
        print(f"{lean_file}: missing preamble target JIT name for {py_file}")
    for py_file, lean_file, target, jit_names in failures:
        print(f"{lean_file}: target JIT `{target}` not found in {py_file}")
        print(f"  python JITs: {jit_names}")
    sys.exit(1)
PY
then
  printf 'ok preamble target JIT scan\n'
else
  printf 'FAIL preamble target JIT scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import ast
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def is_triton_jit(dec: ast.expr) -> bool:
    if isinstance(dec, ast.Call):
        dec = dec.func
    return (
        isinstance(dec, ast.Attribute) and dec.attr == "jit"
    ) or (
        isinstance(dec, ast.Name) and dec.id == "jit"
    )

def python_first_kernel_params(text: str) -> list[str]:
    tree = ast.parse(text)
    lines = text.splitlines()
    preferred = None
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(is_triton_jit(dec) for dec in node.decorator_list):
            continue
        body = "\n".join(lines[node.lineno - 1: node.end_lineno or node.lineno])
        if any(token in body for token in ("tl.program_id", "tl.load", "tl.store")):
            if node.name == "_attn_fwd":
                return [arg.arg for arg in node.args.args]
            if preferred is None:
                preferred = [arg.arg for arg in node.args.args]
    if preferred is not None:
        return preferred
    return []

def lean_first_kernel_signature(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not re.match(r"\s*(?:noncomputable\s+)?def\s+", line):
            continue
        acc = []
        for j in range(i, min(len(lines), i + 60)):
            acc.append(lines[j])
            if "ComputeKernel := triton {" in lines[j]:
                return "\n".join(acc)
    return ""

def lean_binder_groups(signature: str) -> list[str]:
    groups = []
    pairs = {"(": ")", "{": "}", "[": "]"}
    i = 0
    while i < len(signature):
        if signature[i] not in pairs:
            i += 1
            continue
        open_ch = signature[i]
        close_ch = pairs[open_ch]
        depth = 1
        j = i + 1
        while j < len(signature) and depth > 0:
            if signature[j] == open_ch:
                depth += 1
            elif signature[j] == close_ch:
                depth -= 1
            j += 1
        if depth == 0:
            groups.append(signature[i + 1:j - 1])
        i = j
    return groups

def lean_first_kernel_params(text: str) -> list[str]:
    params = []
    for group in lean_binder_groups(lean_first_kernel_signature(text)):
        if ":" not in group:
            continue
        left = group.split(":", 1)[0].strip()
        if not left or left.startswith("h"):
            continue
        params.extend(
            name.lstrip("_")
            for name in re.split(r"\s+", left)
            if name and name != "_"
        )
    return params

def normalize_param(name: str) -> str:
    name = name.lstrip("_").replace("_", "").lower()
    for suffix in ("pointer", "ptr"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if blocker_marker in scope_text:
        continue
    py_params = python_first_kernel_params(py_file.read_text())
    lean_params = lean_first_kernel_params(lean_text)
    lean_param_set = {normalize_param(param) for param in lean_params}
    missing = [param for param in py_params if normalize_param(param) not in lean_param_set]
    if missing:
        failures.append((py_file, lean_file, missing, py_params, lean_params))

if failures:
    for py_file, lean_file, missing, py_params, lean_params in failures:
        print(f"{py_file} -> {lean_file}: Python kernel params missing from Lean")
        print(f"  missing: {missing}")
        print(f"  python:  {py_params}")
        print(f"  lean:    {lean_params}")
    sys.exit(1)
PY
then
  printf 'ok kernel parameter presence scan\n'
else
  printf 'FAIL kernel parameter presence scan\n'
  failures=$((failures + 1))
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
  if awk '/triton \{/{exit} {print}' "${lean_file}" | rg -q 'Translation-surface blocker:'; then
    continue
  fi
  if rg -q 'quant_policy = 0|unquantized path|quant_policy.*4/8' "${lean_file}"; then
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

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def python_first_kernel_body(text: str) -> str:
    lines = text.splitlines()

    def body_for_name(target: str) -> str:
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith("@triton.jit"):
                i += 1
                continue
            i += 1
            while i < len(lines) and lines[i].strip().startswith("@"):
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("def " + target + "("):
                i += 1
                continue

            start = None
            parens = 0
            for j in range(i, len(lines)):
                parens += lines[j].count("(") - lines[j].count(")")
                if parens <= 0 and lines[j].rstrip().endswith(":"):
                    start = j + 1
                    break
            if start is None:
                return ""

            body = []
            k = start
            while k < len(lines):
                line = lines[k]
                if line and not line.startswith((" ", "\t")):
                    break
                body.append(line)
                k += 1
            body_text = "\n".join(body)
            if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
                return body_text
            i = k
        return ""

    preferred = body_for_name("_attn_fwd")
    if preferred:
        return preferred
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue

        def_i = i
        start = None
        parens = 0
        for j in range(i, len(lines)):
            parens += lines[j].count("(") - lines[j].count(")")
            if parens <= 0 and lines[j].rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            return ""

        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
            return body_text
        i = k
    return ""

def lean_first_triton_body(text: str) -> str:
    idx = text.find("triton {")
    if idx < 0:
        return ""
    start = text.find("{", idx)
    depth = 0
    out = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(ch)
    return "".join(out)

def strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

def cast_sequence(text: str) -> list[str]:
    casts = re.findall(r"\.to\(\s*(?:tl\.)?([A-Za-z0-9_\.]+)\s*\)",
        strip_comments(text))
    return [cast.replace("_", "").lower() for cast in casts]

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if blocker_marker in scope_text:
        continue
    py_casts = cast_sequence(python_first_kernel_body(py_file.read_text()))
    lean_casts = cast_sequence(lean_first_triton_body(lean_text))
    if py_casts != lean_casts:
        failures.append((py_file, lean_file, py_casts, lean_casts))

if failures:
    for py_file, lean_file, py_casts, lean_casts in failures:
        print(f"{py_file} -> {lean_file}: .to(...) cast sequence mismatch")
        print(f"  python: {py_casts}")
        print(f"  lean:   {lean_casts}")
    sys.exit(1)
PY
then
  printf 'ok .to(...) cast sequence scan\n'
else
  printf 'FAIL .to(...) cast sequence scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def python_first_kernel_body(text: str) -> str:
    lines = text.splitlines()

    def body_for_name(target: str) -> str:
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith("@triton.jit"):
                i += 1
                continue
            i += 1
            while i < len(lines) and lines[i].strip().startswith("@"):
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("def " + target + "("):
                i += 1
                continue

            start = None
            parens = 0
            for j in range(i, len(lines)):
                parens += lines[j].count("(") - lines[j].count(")")
                if parens <= 0 and lines[j].rstrip().endswith(":"):
                    start = j + 1
                    break
            if start is None:
                return ""

            body = []
            k = start
            while k < len(lines):
                line = lines[k]
                if line and not line.startswith((" ", "\t")):
                    break
                body.append(line)
                k += 1
            body_text = "\n".join(body)
            if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
                return body_text
            i = k
        return ""

    preferred = body_for_name("_attn_fwd")
    if preferred:
        return preferred
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue

        def_i = i
        start = None
        parens = 0
        for j in range(i, len(lines)):
            parens += lines[j].count("(") - lines[j].count(")")
            if parens <= 0 and lines[j].rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            return ""

        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
            return body_text
        i = k
    return ""

def lean_first_triton_body(text: str) -> str:
    idx = text.find("triton {")
    if idx < 0:
        return ""
    start = text.find("{", idx)
    depth = 0
    out = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(ch)
    return "".join(out)

def strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

def tl_calls(text: str, fn: str) -> list[str]:
    text = strip_comments(text)
    pattern = re.compile(rf"tl\s*\.\s*{fn}\s*\(")
    calls = []
    pos = 0
    while True:
        match = pattern.search(text, pos)
        if not match:
            break
        k = match.end()
        depth = 1
        while k < len(text) and depth > 0:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        calls.append(re.sub(r"\s+", " ", text[match.start():k]))
        pos = k
    return calls

def split_top_level_args(call: str, fn: str) -> list[str]:
    match = re.search(rf"tl\s*\.\s*{fn}\s*\(", call)
    inside = call[match.end():-1]
    args = []
    cur = []
    depth = 0
    for ch in inside:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            arg = "".join(cur).strip()
            if arg:
                args.append(arg)
            cur = []
            continue
        cur.append(ch)
    arg = "".join(cur).strip()
    if arg:
        args.append(arg)
    return args

def mask_presence(body: str, fn: str) -> list[bool]:
    out = []
    for call in tl_calls(body, fn):
        args = split_top_level_args(call, fn)
        has_kw = any(re.match(r"^mask\s*=", arg) for arg in args)
        positional = (
            (fn == "load" and len(args) >= 2 and "=" not in args[1])
            or (fn == "store" and len(args) >= 3 and "=" not in args[2])
        )
        out.append(has_kw or positional)
    return out

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if blocker_marker in scope_text:
        continue
    py_body = python_first_kernel_body(py_file.read_text())
    lean_body = lean_first_triton_body(lean_text)
    for fn in ("load", "store"):
        py_masks = mask_presence(py_body, fn)
        lean_masks = mask_presence(lean_body, fn)
        if py_masks != lean_masks:
            failures.append((py_file, lean_file, fn, py_masks, lean_masks))

if failures:
    for py_file, lean_file, fn, py_masks, lean_masks in failures:
        print(f"{py_file} -> {lean_file}: tl.{fn} mask presence mismatch")
        print(f"  python: {py_masks}")
        print(f"  lean:   {lean_masks}")
    sys.exit(1)
PY
then
  printf 'ok tl.load/store mask presence scan\n'
else
  printf 'FAIL tl.load/store mask presence scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def python_first_kernel_body(text: str) -> str:
    lines = text.splitlines()

    def body_for_name(target: str) -> str:
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith("@triton.jit"):
                i += 1
                continue
            i += 1
            while i < len(lines) and lines[i].strip().startswith("@"):
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("def " + target + "("):
                i += 1
                continue

            start = None
            parens = 0
            for j in range(i, len(lines)):
                parens += lines[j].count("(") - lines[j].count(")")
                if parens <= 0 and lines[j].rstrip().endswith(":"):
                    start = j + 1
                    break
            if start is None:
                return ""

            body = []
            k = start
            while k < len(lines):
                line = lines[k]
                if line and not line.startswith((" ", "\t")):
                    break
                body.append(line)
                k += 1
            body_text = "\n".join(body)
            if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
                return body_text
            i = k
        return ""

    preferred = body_for_name("_attn_fwd")
    if preferred:
        return preferred
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue

        def_i = i
        start = None
        parens = 0
        for j in range(i, len(lines)):
            parens += lines[j].count("(") - lines[j].count(")")
            if parens <= 0 and lines[j].rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            return ""

        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
            return body_text
        i = k
    return ""

def lean_first_triton_body(text: str) -> str:
    idx = text.find("triton {")
    if idx < 0:
        return ""
    start = text.find("{", idx)
    depth = 0
    out = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(ch)
    return "".join(out)

def strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

def tl_calls(text: str, fn: str) -> list[str]:
    text = strip_comments(text)
    pattern = re.compile(rf"tl\s*\.\s*{fn}\s*\(")
    calls = []
    pos = 0
    while True:
        match = pattern.search(text, pos)
        if not match:
            break
        k = match.end()
        depth = 1
        while k < len(text) and depth > 0:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        calls.append(text[match.start():k])
        pos = k
    return calls

def normalize_shape_ctor_call(call: str) -> str:
    call = re.sub(r"\$\(([^()]+)\)", r"\1", call)
    call = call.replace("(", "[").replace(")", "]")
    call = call.replace("tl.float32", "float32").replace("tl.float64", "float64")
    call = call.replace("dtype=", "")
    call = re.sub(r"\s+", "", call)
    call = call.replace("_", "").lower()
    call = re.sub(r"\[([^\[\],]+),\]", r"[\1]", call)
    return call

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if blocker_marker in scope_text:
        continue
    py_body = python_first_kernel_body(py_file.read_text())
    lean_body = lean_first_triton_body(lean_text)
    for fn in ("zeros", "full"):
        py_calls = [normalize_shape_ctor_call(call) for call in tl_calls(py_body, fn)]
        lean_calls = [normalize_shape_ctor_call(call) for call in tl_calls(lean_body, fn)]
        if py_calls != lean_calls:
            failures.append((py_file, lean_file, fn, py_calls, lean_calls))

if failures:
    for py_file, lean_file, fn, py_calls, lean_calls in failures:
        print(f"{py_file} -> {lean_file}: tl.{fn} shape/dtype call mismatch")
        print(f"  python: {py_calls}")
        print(f"  lean:   {lean_calls}")
    sys.exit(1)
PY
then
  printf 'ok tl.zeros/full shape dtype scan\n'
else
  printf 'FAIL tl.zeros/full shape dtype scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def python_first_kernel_body(text: str) -> str:
    lines = text.splitlines()

    def body_for_name(target: str) -> str:
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith("@triton.jit"):
                i += 1
                continue
            i += 1
            while i < len(lines) and lines[i].strip().startswith("@"):
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("def " + target + "("):
                i += 1
                continue

            start = None
            parens = 0
            for j in range(i, len(lines)):
                parens += lines[j].count("(") - lines[j].count(")")
                if parens <= 0 and lines[j].rstrip().endswith(":"):
                    start = j + 1
                    break
            if start is None:
                return ""

            body = []
            k = start
            while k < len(lines):
                line = lines[k]
                if line and not line.startswith((" ", "\t")):
                    break
                body.append(line)
                k += 1
            body_text = "\n".join(body)
            if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
                return body_text
            i = k
        return ""

    preferred = body_for_name("_attn_fwd")
    if preferred:
        return preferred
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue

        def_i = i
        start = None
        parens = 0
        for j in range(i, len(lines)):
            parens += lines[j].count("(") - lines[j].count(")")
            if parens <= 0 and lines[j].rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            return ""

        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
            return body_text
        i = k
    return ""

def lean_first_triton_body(text: str) -> str:
    idx = text.find("triton {")
    if idx < 0:
        return ""
    start = text.find("{", idx)
    depth = 0
    out = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(ch)
    return "".join(out)

def strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

def tl_arange_calls(text: str) -> list[str]:
    text = strip_comments(text)
    pattern = re.compile(r"tl\s*\.\s*arange\s*\(")
    calls = []
    pos = 0
    while True:
        match = pattern.search(text, pos)
        if not match:
            break
        k = match.end()
        depth = 1
        while k < len(text) and depth > 0:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        call = re.sub(r"\$\(([^()]+)\)", r"\1", text[match.start():k])
        call = re.sub(r"\s+", "", call)
        calls.append(call.replace("_", "").lower())
        pos = k
    return calls

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if blocker_marker in scope_text:
        continue
    py_calls = tl_arange_calls(python_first_kernel_body(py_file.read_text()))
    lean_calls = tl_arange_calls(lean_first_triton_body(lean_text))
    if py_calls != lean_calls:
        failures.append((py_file, lean_file, py_calls, lean_calls))

if failures:
    for py_file, lean_file, py_calls, lean_calls in failures:
        print(f"{py_file} -> {lean_file}: tl.arange call mismatch")
        print(f"  python: {py_calls}")
        print(f"  lean:   {lean_calls}")
    sys.exit(1)
PY
then
  printf 'ok tl.arange call scan\n'
else
  printf 'FAIL tl.arange call scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def python_first_kernel_body(text: str) -> str:
    lines = text.splitlines()

    def body_for_name(target: str) -> str:
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith("@triton.jit"):
                i += 1
                continue
            i += 1
            while i < len(lines) and lines[i].strip().startswith("@"):
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("def " + target + "("):
                i += 1
                continue

            start = None
            parens = 0
            for j in range(i, len(lines)):
                parens += lines[j].count("(") - lines[j].count(")")
                if parens <= 0 and lines[j].rstrip().endswith(":"):
                    start = j + 1
                    break
            if start is None:
                return ""

            body = []
            k = start
            while k < len(lines):
                line = lines[k]
                if line and not line.startswith((" ", "\t")):
                    break
                body.append(line)
                k += 1
            body_text = "\n".join(body)
            if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
                return body_text
            i = k
        return ""

    preferred = body_for_name("_attn_fwd")
    if preferred:
        return preferred
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue
        start = None
        parens = 0
        for j in range(i, len(lines)):
            parens += lines[j].count("(") - lines[j].count(")")
            if parens <= 0 and lines[j].rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            return ""
        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
            return body_text
        i = k
    return ""

def lean_first_triton_body(text: str) -> str:
    idx = text.find("triton {")
    if idx < 0:
        return ""
    start = text.find("{", idx)
    depth = 0
    out = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(ch)
    return "".join(out)

def strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

def program_id_calls(text: str) -> list[str]:
    text = strip_comments(text)
    pattern = re.compile(r"tl\s*\.\s*program_id\s*\(")
    calls = []
    pos = 0
    while True:
        match = pattern.search(text, pos)
        if not match:
            break
        k = match.end()
        depth = 1
        while k < len(text) and depth > 0:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        call = text[match.end():k - 1]
        call = re.sub(r"\$\(([^()]+)\)", r"\1", call)
        call = re.sub(r"\baxis\s*=", "", call)
        calls.append(re.sub(r"\s+", "", call))
        pos = k
    return calls

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if blocker_marker in scope_text:
        continue
    py_calls = program_id_calls(python_first_kernel_body(py_file.read_text()))
    lean_calls = program_id_calls(lean_first_triton_body(lean_text))
    if py_calls != lean_calls:
        failures.append((py_file, lean_file, py_calls, lean_calls))

if failures:
    for py_file, lean_file, py_calls, lean_calls in failures:
        print(f"{py_file} -> {lean_file}: tl.program_id call mismatch")
        print(f"  python: {py_calls}")
        print(f"  lean:   {lean_calls}")
    sys.exit(1)
PY
then
  printf 'ok tl.program_id call scan\n'
else
  printf 'FAIL tl.program_id call scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def python_first_kernel_body(text: str) -> str:
    lines = text.splitlines()

    def body_for_name(target: str) -> str:
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith("@triton.jit"):
                i += 1
                continue
            i += 1
            while i < len(lines) and lines[i].strip().startswith("@"):
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("def " + target + "("):
                i += 1
                continue

            start = None
            parens = 0
            for j in range(i, len(lines)):
                parens += lines[j].count("(") - lines[j].count(")")
                if parens <= 0 and lines[j].rstrip().endswith(":"):
                    start = j + 1
                    break
            if start is None:
                return ""

            body = []
            k = start
            while k < len(lines):
                line = lines[k]
                if line and not line.startswith((" ", "\t")):
                    break
                body.append(line)
                k += 1
            body_text = "\n".join(body)
            if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
                return body_text
            i = k
        return ""

    preferred = body_for_name("_attn_fwd")
    if preferred:
        return preferred
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue
        start = None
        parens = 0
        for j in range(i, len(lines)):
            parens += lines[j].count("(") - lines[j].count(")")
            if parens <= 0 and lines[j].rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            return ""
        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
            return body_text
        i = k
    return ""

def lean_first_triton_body(text: str) -> str:
    idx = text.find("triton {")
    if idx < 0:
        return ""
    start = text.find("{", idx)
    depth = 0
    out = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(ch)
    return "".join(out)

def strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

def tl_calls(text: str, fn: str) -> list[str]:
    text = strip_comments(text)
    pattern = re.compile(rf"tl\s*\.\s*{fn}\s*\(")
    calls = []
    pos = 0
    while True:
        match = pattern.search(text, pos)
        if not match:
            break
        k = match.end()
        depth = 1
        while k < len(text) and depth > 0:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        calls.append(text[match.end():k - 1])
        pos = k
    return calls

def split_top_level_args(args_text: str) -> list[str]:
    args = []
    cur = []
    depth = 0
    for ch in args_text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            arg = "".join(cur).strip()
            if arg:
                args.append(arg)
            cur = []
            continue
        cur.append(ch)
    arg = "".join(cur).strip()
    if arg:
        args.append(arg)
    return args

def kwarg_sequence(body: str, fn: str) -> list[list[str]]:
    out = []
    for call_args in tl_calls(body, fn):
        kwargs = []
        for arg in split_top_level_args(call_args):
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=", arg)
            if match:
                kwargs.append(match.group(1))
        out.append(kwargs)
    return out

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if blocker_marker in scope_text:
        continue
    py_body = python_first_kernel_body(py_file.read_text())
    lean_body = lean_first_triton_body(lean_text)
    for fn in ("load", "store"):
        py_kwargs = kwarg_sequence(py_body, fn)
        lean_kwargs = kwarg_sequence(lean_body, fn)
        if py_kwargs != lean_kwargs:
            failures.append((py_file, lean_file, fn, py_kwargs, lean_kwargs))

if failures:
    for py_file, lean_file, fn, py_kwargs, lean_kwargs in failures:
        print(f"{py_file} -> {lean_file}: tl.{fn} kwarg sequence mismatch")
        print(f"  python: {py_kwargs}")
        print(f"  lean:   {lean_kwargs}")
    sys.exit(1)
PY
then
  printf 'ok tl.load/store kwarg sequence scan\n'
else
  printf 'FAIL tl.load/store kwarg sequence scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def python_first_kernel_body(text: str) -> str:
    lines = text.splitlines()

    def body_for_name(target: str) -> str:
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith("@triton.jit"):
                i += 1
                continue
            i += 1
            while i < len(lines) and lines[i].strip().startswith("@"):
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("def " + target + "("):
                i += 1
                continue

            start = None
            parens = 0
            for j in range(i, len(lines)):
                parens += lines[j].count("(") - lines[j].count(")")
                if parens <= 0 and lines[j].rstrip().endswith(":"):
                    start = j + 1
                    break
            if start is None:
                return ""

            body = []
            k = start
            while k < len(lines):
                line = lines[k]
                if line and not line.startswith((" ", "\t")):
                    break
                body.append(line)
                k += 1
            body_text = "\n".join(body)
            if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
                return body_text
            i = k
        return ""

    preferred = body_for_name("_attn_fwd")
    if preferred:
        return preferred
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue
        start = None
        parens = 0
        for j in range(i, len(lines)):
            parens += lines[j].count("(") - lines[j].count(")")
            if parens <= 0 and lines[j].rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            return ""
        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
            return body_text
        i = k
    return ""

def lean_first_triton_body(text: str) -> str:
    idx = text.find("triton {")
    if idx < 0:
        return ""
    start = text.find("{", idx)
    depth = 0
    out = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(ch)
    return "".join(out)

def strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

def tl_calls(text: str, fn: str) -> list[str]:
    text = strip_comments(text)
    pattern = re.compile(rf"tl\s*\.\s*{fn}\s*\(")
    calls = []
    pos = 0
    while True:
        match = pattern.search(text, pos)
        if not match:
            break
        k = match.end()
        depth = 1
        while k < len(text) and depth > 0:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        calls.append(text[match.end():k - 1])
        pos = k
    return calls

def split_top_level_args(args_text: str) -> list[str]:
    args = []
    cur = []
    depth = 0
    for ch in args_text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            arg = "".join(cur).strip()
            if arg:
                args.append(arg)
            cur = []
            continue
        cur.append(ch)
    arg = "".join(cur).strip()
    if arg:
        args.append(arg)
    return args

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if blocker_marker in scope_text:
        continue
    py_body = python_first_kernel_body(py_file.read_text())
    lean_body = lean_first_triton_body(lean_text)
    for fn in ("load", "store"):
        py_arg_counts = [len(split_top_level_args(call)) for call in tl_calls(py_body, fn)]
        lean_arg_counts = [len(split_top_level_args(call)) for call in tl_calls(lean_body, fn)]
        if py_arg_counts != lean_arg_counts:
            failures.append((py_file, lean_file, fn, py_arg_counts, lean_arg_counts))

if failures:
    for py_file, lean_file, fn, py_arg_counts, lean_arg_counts in failures:
        print(f"{py_file} -> {lean_file}: tl.{fn} argument count mismatch")
        print(f"  python: {py_arg_counts}")
        print(f"  lean:   {lean_arg_counts}")
    sys.exit(1)
PY
then
  printf 'ok tl.load/store argument count scan\n'
else
  printf 'FAIL tl.load/store argument count scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def python_first_kernel_body(text: str) -> str:
    lines = text.splitlines()

    def body_for_name(target: str) -> str:
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith("@triton.jit"):
                i += 1
                continue
            i += 1
            while i < len(lines) and lines[i].strip().startswith("@"):
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("def " + target + "("):
                i += 1
                continue

            start = None
            parens = 0
            for j in range(i, len(lines)):
                parens += lines[j].count("(") - lines[j].count(")")
                if parens <= 0 and lines[j].rstrip().endswith(":"):
                    start = j + 1
                    break
            if start is None:
                return ""

            body = []
            k = start
            while k < len(lines):
                line = lines[k]
                if line and not line.startswith((" ", "\t")):
                    break
                body.append(line)
                k += 1
            body_text = "\n".join(body)
            if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
                return body_text
            i = k
        return ""

    preferred = body_for_name("_attn_fwd")
    if preferred:
        return preferred
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue
        start = None
        parens = 0
        for j in range(i, len(lines)):
            parens += lines[j].count("(") - lines[j].count(")")
            if parens <= 0 and lines[j].rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            return ""
        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
            return body_text
        i = k
    return ""

def lean_first_triton_body(text: str) -> str:
    idx = text.find("triton {")
    if idx < 0:
        return ""
    start = text.find("{", idx)
    depth = 0
    out = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(ch)
    return "".join(out)

def strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

def tl_calls(text: str, fn: str) -> list[str]:
    text = strip_comments(text)
    pattern = re.compile(rf"tl\s*\.\s*{fn}\s*\(")
    calls = []
    pos = 0
    while True:
        match = pattern.search(text, pos)
        if not match:
            break
        k = match.end()
        depth = 1
        while k < len(text) and depth > 0:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        calls.append(text[match.start():k])
        pos = k
    return calls

def normalize_call(call: str) -> str:
    call = call.replace("\\", "")
    call = re.sub(
        r"\$\(\(([^:()]+)\s*:\s*Region\s*(?:\.[A-Za-z0-9_]+)?\)\)",
        r"\1",
        call,
    )
    call = re.sub(r"\$\(([^()]+)\)", r"\1", call)
    call = call.replace("'", '"')
    call = re.sub(r"(?<![A-Za-z0-9_])0\.(?![0-9])", "0.0", call)
    call = re.sub(r"\(([^()]+)\)\.to\(", r"\1.to(", call)
    call = re.sub(r"mask=\(([^()]+)\)", r"mask=\1", call)
    for dtype in ("float16", "float32", "float64", "int32", "int64", "uint32", "uint64"):
        call = call.replace(f"tl.{dtype}", dtype)
    call = call.replace("True", "true").replace("False", "false")
    call = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", lambda m: m.group(0).replace("_", "").lower(), call)
    call = re.sub(r"\b([A-Za-z0-9]+)(?:ptr|pointer)\b", r"\1", call)
    call = re.sub(r",\s*\)", ")", call)
    call = re.sub(r"\s+", "", call)
    call = re.sub(r"boundarycheck=\(\[([0-9,]+)\]:listnat\)", r"boundarycheck=(\1)", call)
    # The source ports are hand-written Lean syntax, while the Python side
    # comes through AST-preserving text. Treat redundant wrapping parentheses
    # around a whole `mask=(lhs & rhs)` expression as formatting only.
    call = re.sub(r"mask=\(\(([^()]+)\)&(.+)\)\)", r"mask=(\1)&\2))", call)
    call = re.sub(r"mask=\(([^()]+)\)&", r"mask=\1&", call)
    call = re.sub(r"(\[none,:\]\)\))\)+", r"\1", call)
    return call

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if blocker_marker in scope_text:
        continue
    py_body = python_first_kernel_body(py_file.read_text())
    lean_body = lean_first_triton_body(lean_text)
    for fn in ("load", "store"):
        py_calls = [normalize_call(call) for call in tl_calls(py_body, fn)]
        lean_calls = [normalize_call(call) for call in tl_calls(lean_body, fn)]
        if py_calls != lean_calls:
            failures.append((py_file, lean_file, fn, py_calls, lean_calls))

if failures:
    for py_file, lean_file, fn, py_calls, lean_calls in failures:
        print(f"{py_file} -> {lean_file}: tl.{fn} call expression mismatch")
        print(f"  python: {py_calls}")
        print(f"  lean:   {lean_calls}")
    sys.exit(1)
PY
then
  printf 'ok tl.load/store call expression scan\n'
else
  printf 'FAIL tl.load/store call expression scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def python_first_kernel_body(text: str) -> str:
    lines = text.splitlines()

    def body_for_name(target: str) -> str:
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith("@triton.jit"):
                i += 1
                continue
            i += 1
            while i < len(lines) and lines[i].strip().startswith("@"):
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("def " + target + "("):
                i += 1
                continue

            start = None
            parens = 0
            for j in range(i, len(lines)):
                parens += lines[j].count("(") - lines[j].count(")")
                if parens <= 0 and lines[j].rstrip().endswith(":"):
                    start = j + 1
                    break
            if start is None:
                return ""

            body = []
            k = start
            while k < len(lines):
                line = lines[k]
                if line and not line.startswith((" ", "\t")):
                    break
                body.append(line)
                k += 1
            body_text = "\n".join(body)
            if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
                return body_text
            i = k
        return ""

    preferred = body_for_name("_attn_fwd")
    if preferred:
        return preferred
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue

        def_i = i
        start = None
        parens = 0
        for j in range(i, len(lines)):
            parens += lines[j].count("(") - lines[j].count(")")
            if parens <= 0 and lines[j].rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            return ""

        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
            return body_text
        i = k
    return ""

def lean_first_triton_body(text: str) -> str:
    idx = text.find("triton {")
    if idx < 0:
        return ""
    start = text.find("{", idx)
    depth = 0
    out = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(ch)
    return "".join(out)

def strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

def tl_load_calls(text: str) -> list[str]:
    text = strip_comments(text)
    calls = []
    i = 0
    needle = "tl.load("
    while True:
        start = text.find(needle, i)
        if start < 0:
            break
        k = start + len(needle)
        depth = 1
        while k < len(text) and depth > 0:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        calls.append(re.sub(r"\s+", " ", text[start:k]))
        i = k
    return calls

def split_top_level_args(call: str) -> list[str]:
    inside = call[len("tl.load("):-1]
    args = []
    cur = []
    depth = 0
    for ch in inside:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            arg = "".join(cur).strip()
            if arg:
                args.append(arg)
            cur = []
            continue
        cur.append(ch)
    arg = "".join(cur).strip()
    if arg:
        args.append(arg)
    return args

def normalize_other(value: str) -> str:
    value = re.sub(r"\$\(\(([^:()]+)\s*:[^)]+\)\)", r"\1", value.strip())
    value = re.sub(r"\$\(([^()]+)\)", r"\1", value)
    value = value.replace("'", '"')
    if value in ('-inf', 'float("-inf"', '-float("inf"'):
        return "-inf"
    if value in ("0.", "0.0", "0.00"):
        return "0.0"
    return value

def other_values(body: str) -> list[str]:
    out = []
    for call in tl_load_calls(body):
        for arg in split_top_level_args(call):
            if re.match(r"^other\s*=", arg):
                out.append(normalize_other(arg.split("=", 1)[1]))
                break
    return out

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if blocker_marker in scope_text:
        continue
    py_others = other_values(python_first_kernel_body(py_file.read_text()))
    lean_others = other_values(lean_first_triton_body(lean_text))
    if py_others != lean_others:
        failures.append((py_file, lean_file, py_others, lean_others))

if failures:
    for py_file, lean_file, py_others, lean_others in failures:
        print(f"{py_file} -> {lean_file}: tl.load other= mismatch")
        print(f"  python: {py_others}")
        print(f"  lean:   {lean_others}")
    sys.exit(1)
PY
then
  printf 'ok tl.load other literal scan\n'
else
  printf 'FAIL tl.load other literal scan\n'
  failures=$((failures + 1))
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

# Python `+=` coverage. Scoped to `@triton.jit` kernel bodies (host-wrapper
# `+=` statements are not kernel surface), across ALL jit kernels of the file.
if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def python_jit_kernel_bodies(text: str) -> list[str]:
    lines = text.splitlines()
    bodies = []
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue
        start = None
        parens = 0
        for j in range(i, len(lines)):
            parens += lines[j].count("(") - lines[j].count(")")
            if parens <= 0 and lines[j].rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            break
        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        bodies.append("\n".join(body))
        i = k
    return bodies

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    has_iadd = any(
        "+=" in line.split("#", 1)[0]
        for body in python_jit_kernel_bodies(py_file.read_text())
        for line in body.splitlines()
    )
    if not has_iadd:
        continue
    lean_text = lean_file.read_text()
    if "+=" in lean_text:
        continue
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if blocker_marker in scope_text:
        continue
    failures.append(f"{py_file} -> {lean_file}")

if failures:
    for failure in failures:
        print(failure)
    sys.exit(1)
PY
then
  printf 'ok documented += coverage scan\n'
else
  printf 'FAIL Python += missing from Lean without documented slice/scope:\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def norm(name: str) -> str:
    name = name.lower()
    for suffix in ("_pointer", "_ptrs", "_ptr"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name

def python_first_kernel_body(text: str) -> str:
    lines = text.splitlines()

    def body_for_name(target: str) -> str:
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith("@triton.jit"):
                i += 1
                continue
            i += 1
            while i < len(lines) and lines[i].strip().startswith("@"):
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("def " + target + "("):
                i += 1
                continue

            start = None
            parens = 0
            for j in range(i, len(lines)):
                parens += lines[j].count("(") - lines[j].count(")")
                if parens <= 0 and lines[j].rstrip().endswith(":"):
                    start = j + 1
                    break
            if start is None:
                return ""

            body = []
            k = start
            while k < len(lines):
                line = lines[k]
                if line and not line.startswith((" ", "\t")):
                    break
                body.append(line)
                k += 1
            body_text = "\n".join(body)
            if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
                return body_text
            i = k
        return ""

    preferred = body_for_name("_attn_fwd")
    if preferred:
        return preferred
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue
        start = None
        parens = 0
        for j in range(i, len(lines)):
            parens += lines[j].count("(") - lines[j].count(")")
            if parens <= 0 and lines[j].rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            return ""
        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store", "tl.make_block_ptr")):
            return body_text
        i = k
    return ""

def lean_first_triton_body(text: str) -> str:
    idx = text.find("triton {")
    if idx < 0:
        return ""
    start = text.find("{", idx)
    depth = 0
    out = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(ch)
    return "".join(out)

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    py_lhs = {
        norm(match.group(1))
        for match in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\+=",
            python_first_kernel_body(py_file.read_text()), re.M)
    }
    if not py_lhs:
        continue
    lean_text = lean_file.read_text()
    lean_lhs = {
        norm(match.group(1))
        for match in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\+=",
            lean_first_triton_body(lean_text), re.M)
    }
    missing = sorted(py_lhs - lean_lhs)
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if missing and not blocker_marker in scope_text:
        failures.append((py_file, lean_file, missing))

if failures:
    for py_file, lean_file, missing in failures:
        print(f"{py_file} -> {lean_file}: missing += lhs {', '.join(missing)}")
    sys.exit(1)
PY
then
  printf 'ok normalized += lhs coverage scan\n'
else
  printf 'FAIL normalized += lhs coverage scan\n'
  failures=$((failures + 1))
fi

rsqrt_gaps=()
while IFS= read -r py_file; do
  if ! rg -q 'tl(\.math)?\.rsqrt|rsqrt' "${py_file}"; then
    continue
  fi
  dir="${py_file%/*}"
  lean_file="$(find "${dir}" -maxdepth 1 -name '*.lean' | head -n 1)"
  if [ -z "${lean_file}" ]; then
    continue
  fi
  if ! rg -q 'tl(\.math)?\.rsqrt|rsqrt' "${lean_file}"; then
    rsqrt_gaps+=("${py_file} -> ${lean_file}")
  fi
done < <(find "${PORTS_ROOT}" -mindepth 2 -maxdepth 2 -name '*.py' | sort)

if [ "${#rsqrt_gaps[@]}" -gt 0 ]; then
  printf 'FAIL Python rsqrt missing from Lean:\n'
  printf '  %s\n' "${rsqrt_gaps[@]}"
  failures=$((failures + 1))
else
  printf 'ok rsqrt preservation scan\n'
fi

# Lean-only tl.where scan. Operates on comment-stripped Lean code so prose
# mentions of `tl.where` in docstrings cannot trip it (same lesson as the
# placeholder scan), and accepts the standard preamble
# `Translation-surface blocker:` marker exemption like every other textual
# py<->lean scan (a documented proof-slice guard is a declared deviation,
# not an unannotated extra statement).
if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])

def strip_lean_comments(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    depth = 0
    while i < n:
        two = text[i:i + 2]
        if depth == 0 and two == "--":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if two == "/-":
            depth += 1
            out.append("  ")
            i += 2
            continue
        if depth > 0 and two == "-/":
            depth -= 1
            out.append("  ")
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        else:
            out.append(text[i] if text[i] == "\n" else " ")
        i += 1
    return "".join(out)

blocker_marker = "translation-surface blocker:"

failures = []
for lean_file in sorted(root.glob("*/*.lean")):
    lean_text = lean_file.read_text()
    code = strip_lean_comments(lean_text)
    if "tl.where" not in code:
        continue
    py_files = sorted(lean_file.parent.glob("*.py"))
    if not py_files:
        continue
    py_file = py_files[0]
    if "tl.where" in py_file.read_text():
        continue
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if blocker_marker in scope_text:
        continue
    failures.append(f"{lean_file} -> {py_file}")

if failures:
    print("FAIL Lean-only tl.where found:")
    for entry in failures:
        print(f"  {entry}")
    sys.exit(1)
PY
then
  printf 'ok no Lean-only tl.where statements\n'
else
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"
call_re = re.compile(r"\btl(?:\.[A-Za-z_][A-Za-z0-9_]*)+\s*\(")
ignored = {
    "tl.constexpr",
    "tl.float32",
    "tl.float64",
    "tl.int32",
    "tl.int64",
    "tl.uint32",
    "tl.uint64",
    "tl.tensor",
    "tl.for",
}

def python_first_kernel_body(text: str) -> str:
    lines = text.splitlines()

    def body_for_name(target: str) -> str:
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith("@triton.jit"):
                i += 1
                continue
            i += 1
            while i < len(lines) and lines[i].strip().startswith("@"):
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("def " + target + "("):
                i += 1
                continue

            start = None
            parens = 0
            for j in range(i, len(lines)):
                parens += lines[j].count("(") - lines[j].count(")")
                if parens <= 0 and lines[j].rstrip().endswith(":"):
                    start = j + 1
                    break
            if start is None:
                return ""

            body = []
            k = start
            while k < len(lines):
                line = lines[k]
                if line and not line.startswith((" ", "\t")):
                    break
                body.append(line)
                k += 1
            body_text = "\n".join(body)
            if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
                return body_text
            i = k
        return ""

    preferred = body_for_name("_attn_fwd")
    if preferred:
        return preferred
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue

        def_i = i
        start = None
        parens = 0
        for j in range(i, len(lines)):
            parens += lines[j].count("(") - lines[j].count(")")
            if parens <= 0 and lines[j].rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            return ""

        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store", "tl.make_block_ptr")):
            return body_text
        i = k
    return ""

def lean_first_triton_body(text: str) -> str:
    idx = text.find("triton {")
    if idx < 0:
        return ""
    start = text.find("{", idx)
    depth = 0
    out = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(ch)
    return "".join(out)

def strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

def calls(text: str) -> set[str]:
    text = strip_comments(text)
    return {
        match.group(0).split("(", 1)[0].strip()
        for match in call_re.finditer(text)
        if match.group(0).split("(", 1)[0].strip() not in ignored
    }

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    py_calls = calls(python_first_kernel_body(py_file.read_text()))
    lean_text = lean_file.read_text()
    lean_calls = calls(lean_first_triton_body(lean_text))
    missing = sorted(py_calls - lean_calls)
    extra = sorted(lean_calls - py_calls)
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if (missing or extra) and not blocker_marker in scope_text:
        failures.append((py_file, lean_file, missing, extra))

if failures:
    for py_file, lean_file, missing, extra in failures:
        if missing:
            print(f"{py_file} -> {lean_file}: missing tl calls {', '.join(missing)}")
        if extra:
            print(f"{lean_file} -> {py_file}: extra tl calls {', '.join(extra)}")
    sys.exit(1)
PY
then
  printf 'ok tl.* call surface scan\n'
else
  printf 'FAIL tl.* call surface scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def lean_first_preamble(text: str) -> str:
    idx = text.find("triton {")
    return text[:idx] if idx >= 0 else text

def target_kernel_name(preamble: str):
    matches = re.findall(r"\.py`'s[^`]*`([^`]+)`", preamble, re.S)
    return matches[-1] if matches else None

def python_kernel_body(text: str, target) -> str:
    lines = text.splitlines()
    def_i = None
    pending_jit = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("@triton.jit"):
            pending_jit = True
            continue
        if pending_jit and stripped.startswith("def "):
            name = stripped[4:].split("(", 1)[0].strip()
            if target is not None and name != target:
                pending_jit = False
                continue
            def_i = i
            break
        if pending_jit and stripped and not stripped.startswith("@"):
            pending_jit = False
    if def_i is None:
        for i, line in enumerate(lines):
            if line.lstrip().startswith("def "):
                def_i = i
                break
    if def_i is None:
        return text

    start = None
    parens = 0
    for i in range(def_i, len(lines)):
        line = lines[i]
        parens += line.count("(") - line.count(")")
        if parens <= 0 and line.rstrip().endswith(":"):
            start = i + 1
            break
    if start is None:
        return text

    body = []
    for line in lines[start:]:
        if line and not line.startswith((" ", "\t")):
            break
        body.append(line)
    return "\n".join(body)

def lean_triton_body(text: str) -> str:
    idx = text.find("triton {")
    if idx < 0:
        return text
    start = text.find("{", idx)
    depth = 0
    out = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(ch)
    return "".join(out)

def strip_python_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

def python_control_counts(text: str, target) -> tuple[int, int, int]:
    body = strip_python_comments(python_kernel_body(text, target))
    return (
        len(re.findall(r"^\s*for\s+", body, re.M)),
        len(re.findall(r"^\s*while\s+", body, re.M)),
        len(re.findall(r"^\s*if\s+", body, re.M)),
    )

def lean_control_counts(text: str) -> tuple[int, int, int]:
    body = lean_triton_body(text)
    return (
        len(re.findall(r"^\s*(for\s+|tl\.for\s+)", body, re.M)),
        len(re.findall(r"^\s*while\s+", body, re.M)),
        len(re.findall(r"^\s*(if\s+|tl\.if\s+)", body, re.M)),
    )

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    target = target_kernel_name(lean_first_preamble(lean_text))
    py_counts = python_control_counts(py_file.read_text(), target)
    lean_counts = lean_control_counts(lean_text)
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if py_counts != lean_counts and not blocker_marker in scope_text:
        failures.append((py_file, lean_file, py_counts, lean_counts))

if failures:
    for py_file, lean_file, py_counts, lean_counts in failures:
        print(
            f"{py_file} -> {lean_file}: control-flow count mismatch "
            f"python(for,while,if)={py_counts} lean(for,while,if)={lean_counts}"
        )
    sys.exit(1)
PY
then
  printf 'ok kernel control-flow surface scan\n'
else
  printf 'FAIL kernel control-flow surface scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"
call_re = re.compile(r"\btl(?:\.[A-Za-z_][A-Za-z0-9_]*)+\s*\(")
ignored = {
    "tl.constexpr",
    "tl.float32",
    "tl.float64",
    "tl.int32",
    "tl.int64",
    "tl.uint32",
    "tl.uint64",
    "tl.tensor",
    "tl.for",
}

def python_jit_kernel_bodies(text: str) -> list[str]:
    lines = text.splitlines()
    bodies = []
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue

        def_i = i
        start = None
        parens = 0
        for j in range(def_i, len(lines)):
            line = lines[j]
            parens += line.count("(") - line.count(")")
            if parens <= 0 and line.rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            break

        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
            name = lines[def_i].strip().split("def ", 1)[1].split("(", 1)[0]
            if name == "_attn_fwd":
                return [body_text]
            bodies.append(body_text)
        i = k
    return bodies

def lean_triton_bodies(text: str) -> list[str]:
    bodies = []
    pos = 0
    while True:
        idx = text.find("triton {", pos)
        if idx < 0:
            break
        start = text.find("{", idx)
        depth = 0
        out = []
        end = start
        for off, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
                if depth == 1:
                    continue
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = off + 1
                    break
            if depth >= 1:
                out.append(ch)
        bodies.append("".join(out))
        pos = end
    return bodies

def tl_call_sequence(text: str) -> list[str]:
    text = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    out = []
    for match in call_re.finditer(text):
        call = match.group(0).split("(", 1)[0].strip()
        if call not in ignored:
            out.append(call)
    return out

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    py_bodies = python_jit_kernel_bodies(py_file.read_text())
    lean_bodies = lean_triton_bodies(lean_text)
    py_sequences = [tl_call_sequence(py_bodies[0])] if py_bodies else []
    lean_sequences = [tl_call_sequence(lean_bodies[0])] if lean_bodies else []
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if py_sequences != lean_sequences and not blocker_marker in scope_text:
        failures.append((py_file, lean_file, py_sequences, lean_sequences))

if failures:
    for py_file, lean_file, py_sequences, lean_sequences in failures:
        print(f"{py_file} -> {lean_file}: tl call sequence mismatch")
        print(f"  python: {py_sequences}")
        print(f"  lean:   {lean_sequences}")
    sys.exit(1)
PY
then
  printf 'ok tl.* call sequence scan\n'
else
  printf 'FAIL tl.* call sequence scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def python_jit_kernel_bodies(text: str) -> list[str]:
    lines = text.splitlines()
    bodies = []
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue

        def_i = i
        start = None
        parens = 0
        for j in range(i, len(lines)):
            line = lines[j]
            parens += line.count("(") - line.count(")")
            if parens <= 0 and line.rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            break

        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
            name = lines[def_i].strip().split("def ", 1)[1].split("(", 1)[0]
            if name == "_attn_fwd":
                return [body_text]
            bodies.append(body_text)
        i = k
    return bodies

def lean_triton_bodies(text: str) -> list[str]:
    bodies = []
    pos = 0
    while True:
        idx = text.find("triton {", pos)
        if idx < 0:
            break
        start = text.find("{", idx)
        depth = 0
        out = []
        end = start
        for off, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
                if depth == 1:
                    continue
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = off + 1
                    break
            if depth >= 1:
                out.append(ch)
        bodies.append("".join(out))
        pos = end
    return bodies

def norm_name(name: str) -> str:
    name = name.strip().split(":", 1)[0].strip().lower()
    for suffix in ("_pointer", "_ptrs", "_ptr"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name

def split_lhs(lhs: str) -> list[str]:
    parts = [part.strip() for part in lhs.split(",") if part.strip()]
    if not parts:
        return []
    out = []
    for part in parts:
        if part == "_":
            continue
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\s*:[^=]+)?$", part):
            return []
        out.append(norm_name(part))
    return out

def top_assign_lhs(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith(("if ", "elif ")) and ":" in stripped and not stripped.endswith(":"):
        return top_assign_lhs(stripped.split(":", 1)[1])
    if not stripped or stripped.startswith(
        ("tl.store", "tl.atomic", "for ", "if ", "else", "tl.for", "tl.if")
    ):
        return []
    for op in ("+=", "-=", "*=", ":=", "="):
        idx = stripped.find(op)
        if idx <= 0:
            continue
        if op == "=" and (
            stripped[idx - 1 : idx + 1] in ("<=", ">=", "!=", "==") or
            stripped[idx : idx + 2] == "=="
        ):
            continue
        return split_lhs(stripped[:idx])
    return []

def lhs_sequence(text: str) -> list[str]:
    out = []
    depth = 0
    for raw in text.splitlines():
        line = raw.split("#", 1)[0]
        if depth == 0:
            out.extend(top_assign_lhs(line))
        depth += line.count("(") + line.count("[") - line.count(")") - line.count("]")
        depth = max(depth, 0)
    return out

def normalize_guard_lhs_sequence(port: str, seq: list[str]) -> list[str]:
    if port == "bmm_chunk_fwd":
        # Python returns before any side effects for inactive causal blocks.
        # The Lean surface uses the equivalent guard-normal form:
        #   active_block = true
        #   if IS_CAUSAL { active_block = ... }
        #   if active_block { body }
        return [name for name in seq if name != "active_block"]
    return seq

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    py_bodies = python_jit_kernel_bodies(py_file.read_text())
    lean_bodies = lean_triton_bodies(lean_text)
    port = py_file.parent.name
    py_sequences = [normalize_guard_lhs_sequence(port, lhs_sequence(py_bodies[0]))] if py_bodies else []
    lean_sequences = [normalize_guard_lhs_sequence(port, lhs_sequence(lean_bodies[0]))] if lean_bodies else []
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if py_sequences != lean_sequences and not blocker_marker in scope_text:
        failures.append((py_file, lean_file, py_sequences, lean_sequences))

if failures:
    for py_file, lean_file, py_sequences, lean_sequences in failures:
        print(f"{py_file} -> {lean_file}: statement lhs sequence mismatch")
        print(f"  python: {py_sequences}")
        print(f"  lean:   {lean_sequences}")
    sys.exit(1)
PY
then
  printf 'ok statement lhs sequence scan\n'
else
  printf 'FAIL statement lhs sequence scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"
reduce_call_head_re = re.compile(r"\btl\.(sum|max)\s*\(")
axis_kw_re = re.compile(r"(?:^|,)\s*axis\s*=\s*([0-9]+)\s*(?:,|$)")

def python_jit_kernel_bodies(text: str) -> list[str]:
    lines = text.splitlines()
    bodies = []
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue

        def_i = i
        start = None
        parens = 0
        for j in range(i, len(lines)):
            line = lines[j]
            parens += line.count("(") - line.count(")")
            if parens <= 0 and line.rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            break

        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
            name = lines[def_i].strip().split("def ", 1)[1].split("(", 1)[0]
            if name == "_attn_fwd":
                return [body_text]
            bodies.append(body_text)
        i = k
    return bodies

def lean_triton_bodies(text: str) -> list[str]:
    bodies = []
    pos = 0
    while True:
        idx = text.find("triton {", pos)
        if idx < 0:
            break
        start = text.find("{", idx)
        depth = 0
        out = []
        end = start
        for off, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
                if depth == 1:
                    continue
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = off + 1
                    break
            if depth >= 1:
                out.append(ch)
        bodies.append("".join(out))
        pos = end
    return bodies

def logical_lines(text: str) -> list[str]:
    out = []
    cur = ""
    depth = 0
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        cur = (cur + " " + line).strip() if cur else line
        depth += (
            line.count("(") + line.count("[") + line.count("{")
            - line.count(")") - line.count("]") - line.count("}")
        )
        continues = bool(re.search(r"(\+|-|\*|/|&|\||,)$", line))
        if depth <= 0 and not continues:
            out.append(re.sub(r"\s+", " ", cur).strip())
            cur = ""
            depth = 0
    if cur:
        out.append(re.sub(r"\s+", " ", cur).strip())
    return out

def tl_reduce_calls(line: str) -> list[tuple[str, str]]:
    calls = []
    pos = 0
    while True:
        match = reduce_call_head_re.search(line, pos)
        if not match:
            break
        k = match.end()
        depth = 1
        while k < len(line) and depth > 0:
            if line[k] == "(":
                depth += 1
            elif line[k] == ")":
                depth -= 1
            k += 1
        if depth == 0:
            calls.append((match.group(1), line[match.end():k - 1]))
        pos = max(k, match.end())
    return calls

def split_top_level_args(args_text: str) -> list[str]:
    args = []
    cur = []
    depth = 0
    for ch in args_text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            arg = "".join(cur).strip()
            if arg:
                args.append(arg)
            cur = []
            continue
        cur.append(ch)
    arg = "".join(cur).strip()
    if arg:
        args.append(arg)
    return args

def reduce_axis_styles(text: str) -> list[tuple[str, str, str]]:
    styles = []
    for line in logical_lines(text):
        for fn, args in tl_reduce_calls(line):
            axis_kw = axis_kw_re.search(args)
            if axis_kw:
                styles.append((fn, axis_kw.group(1), "kw"))
                continue
            parts = split_top_level_args(args)
            if len(parts) >= 2 and re.fullmatch(r"[0-9]+", parts[1]):
                styles.append((fn, parts[1], "pos"))
    return styles

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    scope_text = lean_text[:lean_text.find("triton {")].lower() if "triton {" in lean_text else lean_text.lower()
    if blocker_marker in scope_text:
        continue
    py_bodies = python_jit_kernel_bodies(py_file.read_text())
    lean_bodies = lean_triton_bodies(lean_text)
    py_styles = reduce_axis_styles(py_bodies[0]) if py_bodies else []
    lean_styles = reduce_axis_styles(lean_bodies[0]) if lean_bodies else []
    if py_styles != lean_styles:
        failures.append((py_file, lean_file, py_styles, lean_styles))

if failures:
    for py_file, lean_file, py_styles, lean_styles in failures:
        print(f"{py_file} -> {lean_file}: reduce axis style mismatch")
        print(f"  python: {py_styles}")
        print(f"  lean:   {lean_styles}")
    sys.exit(1)
PY
then
  printf 'ok reduce axis style scan\n'
else
  printf 'FAIL reduce axis style scan\n'
  failures=$((failures + 1))
fi

# ---- Machine-checkable trust audit (Phase E) --------------------------------
# Two populations, two mechanisms (see documents/TrustAudit.md):
#  * library theorems  -> VeriTile/Meta/TrustReport.lean (#axiomsClean on every
#    proven library theorem). Building the module elaborates every gate;
#    `lake build` of the target both materializes its deps (incl. ApproxGeLU)
#    and fails on any smuggled `sorry`/axiom.
#  * standalone bench   -> bench/audit_trust.sh (external temp-copy #axiomsClean
#    over every bench/tritonbench_g/* port and bench/examples/* file).
# A failure here is a REAL soundness finding (a proof calling itself complete
# while depending on sorryAx / a non-standard axiom), not a scan heuristic.

if lake build VeriTile.Meta.TrustReport; then
  printf 'ok library trust report (#axiomsClean all proven lib theorems)\n'
else
  printf 'FAIL library trust report: a proven library theorem has a bad axiom footprint\n'
  failures=$((failures + 1))
fi

if bench/audit_trust.sh; then
  printf 'ok bench corpus trust audit (#axiomsClean all ports + examples)\n'
else
  printf 'FAIL bench corpus trust audit: a port/example has a bad axiom footprint\n'
  failures=$((failures + 1))
fi

if [ "${failures}" -gt 0 ]; then
  exit 1
fi

printf 'TritonBench-G audit gates passed\n'
