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

known_algorithm_blockers=()

unexpected_algorithm_blockers=()
while IFS= read -r lean_file; do
  if ! rg -q '\(hAlg\s*:' "${lean_file}"; then
    continue
  fi
  rel="${lean_file#${PROJECT_ROOT}/}"
  known=false
  for blocker in "${known_algorithm_blockers[@]}"; do
    if [ "${rel}" = "${blocker}" ]; then
      known=true
      break
    fi
  done
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
    py_sequences = [tl_call_sequence(body) for body in python_jit_kernel_bodies(py_file.read_text())]
    lean_sequences = [tl_call_sequence(body) for body in lean_triton_bodies(lean_text)]
    if py_sequences != lean_sequences and not any(marker in lean_text.lower() for marker in scope_markers):
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

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    py_sequences = [lhs_sequence(body) for body in python_jit_kernel_bodies(py_file.read_text())]
    lean_sequences = [lhs_sequence(body) for body in lean_triton_bodies(lean_text)]
    if py_sequences != lean_sequences and not any(marker in lean_text.lower() for marker in scope_markers):
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
reduce_call_re = re.compile(r"\btl\.(sum|max)\(([^()\n]*(?:\([^()\n]*\)[^()\n]*)*)\)")
axis_kw_re = re.compile(r"(?:^|,)\s*axis\s*=\s*([0-9]+)\s*(?:,|$)")
axis_pos_re = re.compile(r",\s*([0-9]+)\s*(?:,|$)")

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

def reduce_axis_styles(text: str) -> list[tuple[str, str, str]]:
    styles = []
    for line in logical_lines(text):
        for match in reduce_call_re.finditer(line):
            fn = match.group(1)
            args = match.group(2)
            axis_kw = axis_kw_re.search(args)
            if axis_kw:
                styles.append((fn, axis_kw.group(1), "kw"))
                continue
            axis_pos = axis_pos_re.search(args)
            if axis_pos:
                styles.append((fn, axis_pos.group(1), "pos"))
    return styles

failures = []
for py_file in sorted(root.glob("*/*.py")):
    lean_files = sorted(py_file.parent.glob("*.lean"))
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = lean_file.read_text()
    if any(marker in lean_text.lower() for marker in scope_markers):
        continue
    py_styles = [
        style
        for body in python_jit_kernel_bodies(py_file.read_text())
        for style in reduce_axis_styles(body)
    ]
    lean_styles = [
        style
        for body in lean_triton_bodies(lean_text)
        for style in reduce_axis_styles(body)
    ]
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

if [ "${failures}" -gt 0 ]; then
  exit 1
fi

printf 'TritonBench-G audit gates passed\n'
