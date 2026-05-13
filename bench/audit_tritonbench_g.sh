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

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
scope_markers = (
    "slice",
    "outside this",
    "branch",
    "precomputed",
    "surface transcription",
)

def norm(name: str) -> str:
    name = name.lower()
    for suffix in ("_pointer", "_ptrs", "_ptr"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    py_lhs = {
        norm(match.group(1))
        for match in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\+=", py_file.read_text(), re.M)
    }
    if not py_lhs:
        continue
    lean_text = lean_file.read_text()
    lean_lhs = {
        norm(match.group(1))
        for match in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\+=", lean_text, re.M)
    }
    missing = sorted(py_lhs - lean_lhs)
    if missing and not any(marker in lean_text.lower() for marker in scope_markers):
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

lean_only_where=()
while IFS= read -r lean_file; do
  if ! rg -q 'tl\.where' "${lean_file}"; then
    continue
  fi
  dir="${lean_file%/*}"
  py_file="$(find "${dir}" -maxdepth 1 -name '*.py' | head -n 1)"
  if [ -z "${py_file}" ]; then
    continue
  fi
  if ! rg -q 'tl\.where' "${py_file}"; then
    lean_only_where+=("${lean_file} -> ${py_file}")
  fi
done < <(find "${PORTS_ROOT}" -mindepth 2 -maxdepth 2 -name '*.lean' | sort)

if [ "${#lean_only_where[@]}" -gt 0 ]; then
  printf 'FAIL Lean-only tl.where found:\n'
  printf '  %s\n' "${lean_only_where[@]}"
  failures=$((failures + 1))
else
  printf 'ok no Lean-only tl.where statements\n'
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
scope_markers = (
    "slice",
    "outside this",
    "branch",
    "precomputed",
    "surface transcription",
    "single-tile",
    "single-iteration",
    "specializes",
)
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

def calls(text: str) -> set[str]:
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
    py_calls = calls(py_file.read_text())
    lean_text = lean_file.read_text()
    lean_calls = calls(lean_text)
    missing = sorted(py_calls - lean_calls)
    extra = sorted(lean_calls - py_calls)
    if (missing or extra) and not any(marker in lean_text.lower() for marker in scope_markers):
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
scope_markers = (
    "slice",
    "outside this",
    "branch",
    "precomputed",
    "surface transcription",
    "single-tile",
    "single-iteration",
    "specializes",
)

def python_kernel_body(text: str) -> str:
    lines = text.splitlines()
    def_i = None
    pending_jit = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("@triton.jit"):
            pending_jit = True
            continue
        if pending_jit and stripped.startswith("def "):
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

def python_control_counts(text: str) -> tuple[int, int, int]:
    body = strip_python_comments(python_kernel_body(text))
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
        len(re.findall(r"^\s*if\s+", body, re.M)),
    )

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    py_counts = python_control_counts(py_file.read_text())
    lean_text = lean_file.read_text()
    lean_counts = lean_control_counts(lean_text)
    if py_counts != lean_counts and not any(marker in lean_text.lower() for marker in scope_markers):
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

if [ "${failures}" -gt 0 ]; then
  exit 1
fi

printf 'TritonBench-G audit gates passed\n'
