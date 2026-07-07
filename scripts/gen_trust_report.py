#!/usr/bin/env python3
"""Regenerate VeriTile/Meta/TrustReport.lean from scripts/kernel-manifest.tsv.

The trust report is a generated driver module: it imports every library module
that defines a `proven` theorem and runs `#axiomsClean <fully-qualified thm>`
on it. If any proven library theorem's transitive proof smuggles in `sorryAx`
or a non-standard axiom, elaborating this module fails — so

    lake env lean VeriTile/Meta/TrustReport.lean   # exit 0  <=>  all clean

is a single machine-checkable trust gate over the whole library corpus.

Only LIBRARY theorems (file column under `VeriTile/`) go here; standalone
bench ports and the relocated `bench/examples/*` files are not importable and
are audited externally by `bench/audit_trust.sh`.

Usage:  python3 scripts/gen_trust_report.py   (run from repo root)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "scripts/kernel-manifest.tsv"
OUT = ROOT / "VeriTile/Meta/TrustReport.lean"


def module_of(path: str) -> str:
    assert path.endswith(".lean")
    return path[:-len(".lean")].replace("/", ".")


def main() -> None:
    rows = []
    for raw in MAN.read_text().splitlines():
        if raw.startswith("#") or not raw.strip():
            continue
        cols = raw.split("\t")
        if cols[0] == "id":
            continue
        id_, file_, thm, kind, status = cols[0], cols[1], cols[2], cols[3], cols[4]
        if status != "proven":
            continue
        if not file_.startswith("VeriTile/"):
            continue  # bench/examples handled by bench/audit_trust.sh
        rows.append((module_of(file_), thm, id_, kind))

    # group by module, preserve first-seen module order, sort theorems within
    modules = []
    by_mod = {}
    for mod, thm, id_, kind in rows:
        if mod not in by_mod:
            by_mod[mod] = []
            modules.append(mod)
        by_mod[mod].append((thm, id_, kind))
    modules.sort()

    lines = []
    lines.append("/-")
    lines.append("VeriTile.Meta.TrustReport — GENERATED, do not edit by hand.")
    lines.append("")
    lines.append("Regenerate with:  python3 scripts/gen_trust_report.py")
    lines.append("")
    lines.append("Runs `#axiomsClean` on every `proven` LIBRARY theorem in")
    lines.append("scripts/kernel-manifest.tsv. Elaborating this module is the whole-library")
    lines.append("trust gate:")
    lines.append("")
    lines.append("    lake env lean VeriTile/Meta/TrustReport.lean   -- exit 0 <=> all clean")
    lines.append("")
    lines.append("A failure means a `proven` theorem's transitive proof depends on `sorryAx`")
    lines.append("or a non-standard axiom (footprint outside {propext, Classical.choice,")
    lines.append("Quot.sound}) — a real soundness finding, NOT something to paper over.")
    lines.append("")
    lines.append(f"Coverage: {len(rows)} library theorems across {len(modules)} modules.")
    lines.append("Standalone bench ports + bench/examples are audited by bench/audit_trust.sh.")
    lines.append("-/")
    lines.append("import VeriTile.Meta.StatementAudit")
    for mod in modules:
        lines.append(f"import {mod}")
    lines.append("")
    lines.append("-- `#axiomsClean` is a global command registered by importing")
    lines.append("-- VeriTile.Meta.StatementAudit; fully-qualified names resolve without `open`.")
    lines.append("")
    for mod in modules:
        lines.append(f"-- {mod}")
        for thm, id_, kind in sorted(by_mod[mod]):
            lines.append(f"#axiomsClean {thm}")
        lines.append("")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} theorems, {len(modules)} modules")


if __name__ == "__main__":
    main()
