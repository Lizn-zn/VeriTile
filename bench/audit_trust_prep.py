#!/usr/bin/env python3
"""Emit an audited temp copy of a standalone bench Lean file.

Given a path to a bench Lean file (a `bench/tritonbench_g/*/*.lean` port, a
`bench/examples/*.lean` showcase, or a `bench/tests/*.lean` smoke), print to
stdout a copy that:

  1. adds `import VeriTile.Meta.StatementAudit` to the import block, and
  2. appends, at end of file, a `#axiomsClean <fully-qualified-thm>` command for
     every headline theorem, plus Lean's `#auditModuleSpecs` discovery gate.

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

# These fixtures deliberately contain rejected specs inside #guard_msgs.
# Their own commands assert those failures; re-auditing the final fixture
# environment would reject the intentionally retained negative examples.
NEGATIVE_SPEC_FIXTURES = {
    "bench/tests/StatementAudit.lean",
    "bench/tests/ModuleSpecAudit.lean",
}

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


def parse_decls(lines):
    """Yield declaration names for axiom checks; kernel types are read by Lean."""
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
            out.append((kind, name, fq))
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
    for kind, name, fq in decls:
        if kind == "specification" or (
            kind in ("theorem", "lemma") and name.endswith(HEADLINE_SUFFIXES)):
            headline.append(fq)
    # plus fully-qualified manifest names for this file
    headline.extend(manifest_names_for(rel))
    # dedupe, keep order
    seen = set()
    axioms_targets = [n for n in headline if not (n in seen or seen.add(n))]

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
    # Lean discovers definitions by elaborated result type, rather than a
    # first-line regex that mistakes `(kernel : ComputeKernel)` parameters
    # for return types. The command prints its actual coverage inventory.
    if rel in NEGATIVE_SPEC_FIXTURES:
        footer.append('#eval IO.println "Spec audit: negative-test fixture; guarded checks executed in source"')
    else:
        footer.append("#auditModuleSpecs")

    sys.stdout.write("\n".join(new_lines + footer) + "\n")

    # a manifest of what we audited, to stderr (for the driver's diagnostics)
    sys.stderr.write(f"{rel}\taxiomsClean={len(axioms_targets)}\tspecNonCircular=Lean-environment\n")


if __name__ == "__main__":
    main()
