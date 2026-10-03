#!/usr/bin/env python3
"""Replay the three primitive probes and maintain their current combined table."""
import argparse
from collections import Counter
from pathlib import Path
import tempfile

if __package__:
    from . import check_numerics_supplement as runner, report_numerics as reporting
else:
    import check_numerics_supplement as runner
    import report_numerics as reporting


def publish(run, output):
    status = runner.read_json(run / "status.json")
    if status["Status"] != "Succeeded" or (run / "exit_code").read_text().strip() != "0":
        raise ValueError("the DLC job must finish successfully before publication")
    expected = {("EXP-SUB-INTRINSIC", "fp32"), ("COUNT-ZERO", "int32_fp32"),
                ("COUNT-SUCCESSOR", "int32_fp32")}
    rows, manifests, admissions = [], {}, {}
    with tempfile.TemporaryDirectory(dir=run, prefix="cpu-replay-") as temporary:
        staging = Path(temporary)
        for name in ("exp", "counts"):
            bundle = run / name / "atomic"
            replay = runner.replay(bundle)
            runner.publish_report(bundle, replay, staging / name)
            for filename in ("summary.json", "summary.csv", "summary.md"):
                if (staging / name / filename).read_bytes() != (run / name / "report" / filename).read_bytes():
                    raise ValueError(f"GPU/CPU table mismatch: {name}/{filename}")
            table = runner.read_json(staging / name / "summary.json")
            if any(r["state"] != "COMPLETE" or not r["replayed"] for r in table["rows"]):
                raise ValueError("all three requested probes must complete and replay")
            rows.extend(table["rows"])
            manifests[name] = runner.read_json(bundle / "manifest.json")
            admissions[name] = replay
    if len(rows) != 3 or {(r["rule"], r["format"]) for r in rows} != expected:
        raise ValueError("unexpected primitive report coverage")
    boundaries = runner.read_json(run / "count-boundaries.json")
    if (boundaries["sources"] != runner.source_hashes()
            or boundaries["audit_source_sha256"] != runner.sha((runner.ROOT / "scripts/check_count_boundaries.py").read_bytes())
            or boundaries["exhaustive"] != {"low": 0, "high_exclusive": 2**24,
                                            "tested": 2**24, "mismatches": 0}):
        raise ValueError("missing or inconsistent exhaustive count check")
    for row in rows:
        if row["rule"] == "COUNT-ZERO":
            row["reason"] = "constant zero conversion; repeated execution does not enlarge its domain"
        elif row["rule"] == "COUNT-SUCCESSOR":
            row["reason"] = "integer 0 <= i < 16777216 only; out-of-range equality is false"
    table = {"rows": rows, "total": 3, "accepted": sum(r["accept"] for r in rows),
             "replayed": 3, "complete": True, "states": dict(Counter(r["state"] for r in rows)),
             **reporting.DEFINITIONS}
    reporting.publish(output, table)
    runner.write_json(output / "experiment.json", {
        "job": {k: status.get(k) for k in ("JobId", "DisplayName", "Status", "GmtRunningTime", "GmtFinishTime")},
        "hardware": {"model": "H200", "runtime_name": boundaries["hardware"]},
        "independent_cpu_replay": "all three summary tables exactly match GPU-environment replay",
        "bundles": manifests,
    })
    runner.write_json(output / "admission.json", admissions)
    runner.write_json(output / "count-boundaries.json", boundaries)
    print(f"Replayed all 3 primitives; {table['accepted']} accepted. Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=runner.ROOT / "experiments/floating_point/primitives/report")
    args = parser.parse_args()
    publish(args.run, args.output)


if __name__ == "__main__":
    main()
