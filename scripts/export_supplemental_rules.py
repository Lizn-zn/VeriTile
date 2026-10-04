#!/usr/bin/env python3
"""Import the current user-trusted report as guarded Lean scalar-rule data.

No GPU replay and no EvidenceValidated proofs are manufactured. Reported WARN,
domain events, unsupported combinations and smoke results cannot be promoted.
"""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import re

if __package__:
    from . import check_numerics_supplement as experiment
    from .export_numerical_rules import (lean_string, declaration_name, validate_accepted_bounds,
                                         load_report as load_original_report, REPORT as ORIGINAL_REPORT)
else:
    import check_numerics_supplement as experiment
    from export_numerical_rules import (lean_string, declaration_name, validate_accepted_bounds,
                                        load_report as load_original_report, REPORT as ORIGINAL_REPORT)

ROOT = experiment.ROOT
REPORT = ROOT / "experiments/floating_point/supplement/report"
OUTPUT = ROOT / "VeriTile/Triton/Float/SupplementalAdmission.lean"
# Exact source identity of the already trusted PR #11 report, before filtering.
LEGACY_SOURCE_SNAPSHOT = "22154c3275d43497000aed762e953fd6a83d40f3810404d8cf470f5c4c881545"


def domain(rule):
    if rule in experiment.supplemental.COUNT_RULES:
        raise ValueError("count-conversion evidence requires an integer-range-aware Lean binding")
    catalog = experiment.supplemental.load_catalog()
    guards = [(name, "finite") for name in catalog[rule]["operands"]]
    if rule == "DIV-MUL-RCP":
        guards.append(("b", "nonzero"))
    elif rule == "MUL-RCP-CANCEL":
        guards.append(("a", "nonzero"))
    elif rule in {"LOG-MUL", "LOG-MUL-LIBDEVICE"}:
        guards.extend([(name, "positive") for name in ("a", "b")])
    return guards


def load_report(directory):
    settings = experiment.read_json(directory / "experiment.json")
    summary = experiment.read_json(directory / "summary.json")
    profile = experiment.validate_profile(deepcopy(settings["profile"]))
    legacy = (experiment.sha(experiment.original.registry.canonical_json(settings["sources"]))
              == LEGACY_SOURCE_SNAPSHOT)
    if settings["sources"] != experiment.source_hashes() and not legacy:
        raise ValueError("report source hashes differ from the numerical implementation")
    expected_version = "scalar-supplement-5" if legacy else experiment.BUNDLE_VERSION
    if settings["bundle_version"] != expected_version or settings["smoke"] is not False:
        raise ValueError("only formal supplemental bundles can be imported")
    if profile["gates"]["warning_policy"] != "pass_only":
        raise ValueError("this exporter requires pass_only admission")
    manifest = {k: v for k, v in settings.items() if k != "execution"}
    canonical = experiment.original.registry.canonical_json
    if experiment.sha(canonical(manifest) + b"\n") != settings["execution"]["manifest_sha256"]:
        raise ValueError("experiment manifest hash mismatch")
    formats = {f["name"]: f for f in profile["formats"]}
    expected = {(r, f) for r in profile["rules"] for f in formats}
    if settings["entries"] != [f"{f}__{r}" for f in formats for r in profile["rules"]]:
        raise ValueError("manifest entries disagree with profile")
    seen, accepted = set(), []
    for row in summary["rows"]:
        key = row["rule"], row["format"]
        if key not in expected or key in seen:
            raise ValueError("unknown or duplicate rule/format row")
        seen.add(key)
        if any(type(row[k]) is not bool for k in ("accept", "replayed")):
            raise ValueError("published report requires final Boolean flags")
        passed = (row["state"] == "COMPLETE" and row["replayed"]
                  and row["decision"] == "ACCEPT" and row["bias"] == row["vars"] == "PASS")
        if row["accept"] != passed:
            raise ValueError("accept disagrees with the reported state and gates")
        if row["state"] == "COMPLETE":
            if experiment.supplemental.unsupported(key[0], formats[key[1]]):
                raise ValueError("unsupported precision cannot report COMPLETE")
            count = row["replicates"]
            cap = ((profile["replicates_max"] + profile["batch"] - 1) // profile["batch"]) * profile["batch"]
            if (type(count) is not int or not profile["replicates"] <= count <= cap
                    or count % profile["batch"] or row["replayed"] is not True):
                raise ValueError("invalid complete replicate count or replay flag")
        elif row["state"] not in {"NUMERIC_EVENT", "UNSUPPORTED"} or row["replayed"]:
            raise ValueError("report has unfinished or inconsistent rows")
        if passed:
            validate_accepted_bounds(row, profile)
            accepted.append(row)
    if (seen != expected or summary["total"] != len(expected)
            or summary["accepted"] != len(accepted)
            or summary["replayed"] != sum(r["replayed"] for r in summary["rows"])
            or summary["states"] != dict(Counter(r["state"] for r in summary["rows"]))):
        raise ValueError("inconsistent report coverage or totals")
    hashes = {name: experiment.sha((directory / name).read_bytes())
              for name in ("experiment.json", "summary.json")}
    snapshot = experiment.sha(canonical(hashes))
    return settings, profile, accepted, hashes, snapshot


def render(directory=REPORT, namespace="SupplementalAdmission"):
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", namespace) is None:
        raise ValueError("namespace must be a single Lean identifier")
    settings, profile, rows, hashes, snapshot = load_report(directory)
    original_count = len(load_original_report(ORIGINAL_REPORT)[2])
    canonical = experiment.original.registry.canonical_json
    try:
        path = str(directory.resolve().relative_to(ROOT))
    except ValueError:
        path = str(directory.resolve())
    metadata = {"trust": "user-trusted published report; no independent replay during export",
                "report_sha256": hashes, "experiment": settings}
    encoded = canonical(metadata).decode("ascii")
    lines = ["/- Generated by scripts/export_supplemental_rules.py --trust-report.",
             "   Accepted rows only; domains and precision remain part of each atom.",
             "   This file contains data, not numerical validity proofs or IEEE axioms. -/",
             "import VeriTile.Triton.Float.GuardedRules",
             "import VeriTile.Triton.Float.ReportedAdmission", "",
             f"namespace VeriTile.Triton.FP.{namespace}", "",
             f"def snapshot : String := {lean_string(snapshot)}",
             f"def reportMetadata : Lean.Json := (Lean.Json.parse {lean_string(encoded)}).toOption.getD .null", ""]
    formats = {f["name"]: f for f in profile["formats"]}
    for row in rows:
        fmt, rule = formats[row["format"]], row["rule"]
        guards = ", ".join(f"⟨{lean_string(n)}, .{k}⟩" for n, k in domain(rule))
        payload = {"row": row, "relation": experiment.supplemental.load_catalog()[rule]}
        encoded_row = canonical(payload).decode("ascii")
        key = f"report:{snapshot}:{row['format']}:{rule}"
        artifact = f"{path}/summary.json#{row['format']}/{rule}"
        lines += [f"def {declaration_name(row)} : ReportedScalarRule where",
                  "  report := {",
                  f"    ruleID := {lean_string(rule)}",
                  f"    format := {lean_string(row['format'])}",
                  f"    key := {lean_string(key)}",
                  f"    artifact := {lean_string(artifact)}",
                  "    configuration := Lean.Json.mkObj [(\"report\", reportMetadata),",
                  f"      (\"relation\", (Lean.Json.parse {lean_string(encoded_row)}).toOption.getD .null)]",
                  f"    description := {lean_string(experiment.supplemental.load_catalog()[rule]['domain'])}",
                  f"    shape := {profile['shape']}", f"    block := {profile['launch']['block']}"]
        lines += [f"    {k} := {lean_string(fmt[k])}" for k in ("input", "compute", "accumulator", "output")]
        lines += ["  }", f"  guards := [{guards}]", ""]
    lines += ["def all : List ReportedScalarRule := [",
              "  " + ",\n  ".join(declaration_name(r) for r in rows) + "]", "",
              f"theorem accepted_count : all.length = {len(rows)} := rfl"]
    if namespace == "SupplementalAdmission":
        lines += [f"theorem total_accepted_count : ReportedAdmission.all.length + all.length = {original_count + len(rows)} := rfl"]
    lines += ["", f"end VeriTile.Triton.FP.{namespace}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--namespace", default="SupplementalAdmission")
    parser.add_argument("--trust-report", action="store_true", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        content = render(args.report, args.namespace)
        if args.check:
            if args.output.read_text() != content:
                raise ValueError("generated supplemental table is stale")
        else:
            args.output.write_text(content)
        print(f"{'Checked' if args.check else 'Generated'} {args.output}")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
