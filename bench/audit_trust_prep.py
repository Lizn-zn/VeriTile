#!/usr/bin/env python3
"""Emit an audited temp copy of a standalone bench Lean file.

Given a path to a bench Lean file (a `bench/tritonbench_g/*/*.lean` port or a
`bench/examples/*.lean` file), print to stdout a copy that:

  1. adds `import VeriTile.Meta.StatementAudit` to the import block, and
  2. appends, at end of file, a `#axiomsClean <fully-qualified-thm>` command for
     every headline theorem, plus `#specNonCircular` for discoverable specs.

Compiling the emitted copy with `lake env lean` therefore both re-checks the
port AND runs the trust audit — an external gate that never touches the port
files themselves. `bench/audit_trust.sh` drives this per file, in parallel.

Headline theorems are discovered by the uniform bench naming convention
(`*_correct`, `*_compute_correct`, `*_output_summary`, `*_output_summary_general`)
and, for manifested files, by the fully-qualified names in
scripts/kernel-manifest.tsv. Names are emitted fully-qualified (namespace-aware)
so they resolve regardless of `open`s.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts/kernel-manifest.tsv"

HEADLINE_SUFFIXES = (
    "_compute_correct",
    "_correct",
    "_output_summary_general",
    "_output_summary",
)

def strip_lean_comments(text: str) -> str:
    """Blank out `--` line comments and (nested) `/- ... -/` block comments,
    preserving line structure, so keyword-at-line-start decl scans cannot
    match prose (e.g. a docstring line starting with "specification of...")."""
    out: list[str] = []
    depth = 0
    i, n = 0, len(text)
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
        if depth > 0:
            out.append("\n" if text[i] == "\n" else " ")
            i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


decl_re = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+|scoped\s+)*"
    r"(theorem|specification|lemma|def|denotation|abbrev)\s+([A-Za-z_][A-Za-z0-9_'\.]*)"
)
ns_re = re.compile(r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_'\.]*)")
end_named_re = re.compile(r"^\s*end\s+([A-Za-z_][A-Za-z0-9_'\.]*)\s*$")
end_bare_re = re.compile(r"^\s*end\s*$")
sec_re = re.compile(r"^\s*section\b")
kernel_re = re.compile(r":\s*ComputeKernel\b")


def parse_decls(lines):
    """Yield (kind, bare_name, fq_name, is_kernel) tracking namespace nesting."""
    stack = []  # entries: ('ns', name) or ('sec', None/name)
    out = []
    for line in lines:
        if ns_re.match(line):
            stack.append(("ns", ns_re.match(line).group(1)))
            continue
        if sec_re.match(line):
            stack.append(("sec", None))
            continue
        if end_named_re.match(line) or end_bare_re.match(line):
            if stack:
                stack.pop()
            continue
        m = decl_re.match(line)
        if m:
            kind, name = m.group(1), m.group(2)
            nsparts = [s[1] for s in stack if s[0] == "ns"]
            fq = ".".join(nsparts + [name]) if nsparts else name
            is_kernel = bool(kernel_re.search(line))
            out.append((kind, name, fq, is_kernel))
    return out


def manifest_names_for(rel_path):
    names = []
    if not MANIFEST.exists():
        return names
    for raw in MANIFEST.read_text().splitlines():
        if raw.startswith("#") or not raw.strip():
            continue
        cols = raw.split("\t")
        if cols[0] == "id" or len(cols) < 5:
            continue
        if cols[1] == rel_path and cols[4] == "proven":
            names.append(cols[2])
    return names


def main():
    path = Path(sys.argv[1]).resolve()
    rel = str(path.relative_to(ROOT))
    text = path.read_text()
    lines = text.splitlines()

    decls = parse_decls(strip_lean_comments(text).splitlines())

    # headline theorems: every `specification` decl (the keyword marks the
    # public surface), plus the legacy suffix net (kept as a superset so the
    # gate only ever widens, never narrows)
    headline = []
    for kind, name, fq, _ in decls:
        if kind == "specification" or (
            kind in ("theorem", "lemma") and name.endswith(HEADLINE_SUFFIXES)):
            headline.append(fq)
    # plus fully-qualified manifest names for this file
    headline.extend(manifest_names_for(rel))
    # dedupe, keep order
    seen = set()
    axioms_targets = [n for n in headline if not (n in seen or seen.add(n))]

    # spec / kernel discovery for #specNonCircular
    specs = [fq for kind, name, fq, _ in decls
             if kind in ("def", "abbrev") and name.endswith("Spec")]
    kernels = [fq for kind, name, fq, isk in decls if kind == "def" and isk]

    # ---- build the temp copy: insert import after the import block ----
    last_import = -1
    for i, line in enumerate(lines):
        if line.startswith("import "):
            last_import = i
    new_lines = list(lines)
    audit_import = "import VeriTile.Meta.StatementAudit"
    if audit_import not in new_lines:
        insert_at = last_import + 1 if last_import >= 0 else 0
        new_lines.insert(insert_at, audit_import)

    footer = ["", "-- ==== external trust audit (appended by bench/audit_trust.sh) ===="]
    for t in axioms_targets:
        footer.append(f"#axiomsClean {t}")
    if specs and kernels:
        klist = ", ".join(kernels)
        for s in specs:
            footer.append(f"#specNonCircular {s} avoiding [{klist}]")

    sys.stdout.write("\n".join(new_lines + footer) + "\n")

    # a manifest of what we audited, to stderr (for the driver's diagnostics)
    sys.stderr.write(f"{rel}\taxiomsClean={len(axioms_targets)}\tspecNonCircular={len(specs) if (specs and kernels) else 0}\n")


if __name__ == "__main__":
    main()
