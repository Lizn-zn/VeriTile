#!/usr/bin/env python3
"""Run/replay supplemental scalar atoms using the shared local-ULP two gates."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time

import numpy as np

if __package__:
    from . import check_numerics as original, supplement_numerics as supplemental
else:
    import check_numerics as original
    import supplement_numerics as supplemental

ROOT = original.ROOT
DEFAULT_PROFILE = supplemental.DEFAULT_PROFILE
KERNELS = supplemental.KERNELS
SOURCES = [*original.SOURCES, Path(__file__).resolve(), Path(supplemental.__file__),
           KERNELS, supplemental.CATALOG]
ACCEPTED = original.ACCEPTED
BUNDLE_VERSION = "scalar-supplement-7"
gates = original.gates
domains = original.domains
NumericEvent = original.NumericEvent
sha, write_json, read_json = original.sha, original.write_json, original.read_json
load_module, seed_for, shapes_for = original.load_module, original.seed_for, original.shapes_for
validate_profile, contract_for = supplemental.validate_profile, supplemental.contract_for
oracle, launch_pair = supplemental.oracle, supplemental.launch_pair


def source_hashes():
    return {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in SOURCES}


def run_instance(torch, triton, kernels, profile, fmt, rule, directory, backend, sources, smoke):
    directory.mkdir(exist_ok=True)
    record_path = directory / "record.json"
    base = {"rule_id": rule, "format": fmt["name"], "state": "RUNNING", "decision": "NOT_EVALUATED"}
    write_json(record_path, base)
    if reason := supplemental.unsupported(rule, fmt):
        write_json(record_path, {**base, "state": "UNSUPPORTED", "decision": "UNSUPPORTED", "reason": reason})
        return "UNSUPPORTED"
    observations = {key: [] for key in gates.OBSERVATIONS}
    valid_counts = []
    seed = seed_for(profile, fmt, rule)
    lowerings = None
    started = time.monotonic()
    try:
        stop = None
        generator = torch.Generator(device="cuda").manual_seed(seed)
        limit = ((profile["replicates_max"] + profile["batch"] - 1) // profile["batch"]) * profile["batch"]
        for replicate in range(limit):
            inputs = supplemental.sample_inputs(torch, profile, fmt, rule, generator)
            valid = domains.mask(torch, rule, inputs)
            valid_counts.append(int(valid.sum().item()))
            if valid_counts[-1] == 0:
                continue
            exact = oracle(torch, rule, inputs)
            (reference, candidate), compiled, errors = launch_pair(torch, triton, kernels, rule, inputs, profile, fmt)
            if lowerings is None:
                lowerings = {}
                for side, programs in compiled.items():
                    lowerings[side] = []
                    for index, ptx in enumerate(programs):
                        name = f"{side}.{index}.ptx"
                        (directory / name).write_text(ptx)
                        lowerings[side].append(sha(ptx.encode()))
                config = contract_for(profile, fmt, rule, backend, sources, lowerings)
                base.update(config=config, instance_key=supplemental.instance_key(config), lowerings=lowerings)
                write_json(record_path, base)
            elif any([sha(p.encode()) for p in compiled[side]] != lowerings[side] for side in lowerings):
                raise ValueError("compiled lowering changed within one instance")
            sample = supplemental.observe(torch, reference, candidate, exact, fmt, errors, valid)
            for key in observations:
                observations[key].append(sample[key])
            if replicate == 0 or (replicate + 1) % 32 == 0:
                print(f"  {fmt['name']} {rule}: {replicate + 1}/{limit} ({time.monotonic() - started:.1f}s)", flush=True)
            if len(observations["delta"]) % profile["batch"] == 0:
                var, stop = gates.checkpoint(observations, profile["gates"]["vars"],
                                             profile["replicates"], profile["replicates_max"], profile["batch"])
                print(f"  magnitude: {var['status']} U={var['upper']} stop={stop}", flush=True)
                if stop:
                    break
        if stop is None:
            raise NumericEvent("INCONCLUSIVE", "insufficient nonempty replicates within the draw budget")
        arrays = {key: np.asarray(values, dtype=np.float64) for key, values in observations.items()}
        count = len(arrays["delta"])
        result = gates.evaluate(arrays, profile["gates"], seed, count, smoke)
        result.update(stopping_reason=stop, completed_replicates=count)
        np.savez_compressed(directory / "observations.npz", **arrays,
                            valid_samples=np.asarray(valid_counts, dtype=np.int64))
        base.update(state="COMPLETE", result=result, decision=result["decision"],
                    observations_sha256=sha((directory / "observations.npz").read_bytes()),
                    seconds=time.monotonic() - started)
    except NumericEvent as event:
        base.update(state="NUMERIC_EVENT", decision="SMOKE_ONLY" if smoke else event.status,
                    reason=str(event), completed_replicates=len(observations["delta"]))
    except Exception as error:
        base.update(state="ERROR", decision="NOT_EVALUATED", reason=f"{type(error).__name__}: {error}",
                    completed_replicates=len(observations["delta"]))
    base["sampling"] = domains.summary(valid_counts, profile["shape"])
    write_json(record_path, base)
    return base["decision"]


def run(args):
    profile = validate_profile(deepcopy(load_module(args.profile.resolve()).PROFILE))
    if args.rules:
        profile["rules"] = args.rules.split(",")
    if args.formats:
        requested = args.formats.split(",")
        known = {fmt["name"] for fmt in profile["formats"]}
        if not set(requested) <= known:
            raise ValueError("unknown --formats entry")
        profile["formats"] = [f for f in profile["formats"] if f["name"] in requested]
    if args.smoke:
        profile["shape"] = [32, 33]
        profile["replicates"] = profile["replicates_max"] = profile["batch"] = 4
    validate_profile(profile)
    try:
        import torch
        import triton
    except ImportError as error:
        raise ValueError("GPU run requires torch and triton in your CUDA environment; see experiments/floating_point/supplement/README.md") from error
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise ValueError("this runner requires an NVIDIA CUDA GPU; no CPU numerical fallback is used")
    kernels = load_module(KERNELS)
    if kernels.SUPPORTED != set(supplemental.load_catalog()):
        raise ValueError("candidate catalogue and executable templates disagree")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True, timeout=10).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        driver = ["unavailable"]
    backend = {"kind": "triton-cuda", "implementation_version": source_hashes()[str(KERNELS.relative_to(ROOT))],
               "target": {"device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                          "driver_versions": driver},
               "compiler": {"torch": str(torch.__version__), "triton": triton.__version__, "cuda": torch.version.cuda,
                            "numpy": np.__version__},
               "compile_options": {"enable_fp_fusion": False},
               "launch": profile["launch"]}
    manifest = {"bundle_version": BUNDLE_VERSION, "profile": profile, "smoke": args.smoke,
                "sources": source_hashes(), "backend": backend,
                "entries": [f"{f['name']}__{r}" for f in profile["formats"] for r in profile["rules"]]}
    if args.resume:
        old = read_json(args.output / "manifest.json")
        if old != manifest:
            raise ValueError("resume requires identical configuration, source, device and software versions")
        replay(args.output)  # Verify retained records before skipping them.
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "manifest.json", manifest)
        source_dir = args.output / "sources"
        source_dir.mkdir()
        for path in SOURCES:
            saved = source_dir / path.relative_to(ROOT)
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(path.read_bytes())
        (source_dir / "config.py").write_bytes(args.profile.read_bytes())
    errors = 0
    for fmt in profile["formats"]:
        for rule in profile["rules"]:
            directory = args.output / f"{fmt['name']}__{rule}"
            record = directory / "record.json"
            if args.resume and record.exists() and read_json(record)["state"] in ("COMPLETE", "UNSUPPORTED", "NUMERIC_EVENT"):
                print(f"[KEEP] {directory.name}", flush=True)
                continue
            print(f"[RUN] {directory.name}", flush=True)
            decision = run_instance(torch, triton, kernels, profile, fmt, rule, directory,
                                    backend, manifest["sources"], args.smoke)
            print(f"[{decision}] {directory.name}", flush=True)
            errors += read_json(record)["state"] == "ERROR"
            torch.cuda.empty_cache()
    print(f"Bundle saved to {args.output}; return the entire directory, including observations and PTX.")
    return 1 if errors else 0


def replay(bundle):
    """Read only JSON/NPZ/PTX. Never execute a returned profile or Python source."""
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("bundle_version") != BUNDLE_VERSION or type(manifest.get("smoke")) is not bool:
        raise ValueError("unsupported bundle schema")
    if manifest["sources"] != source_hashes():
        raise ValueError("source hashes differ: check out the exact experiment revision before import")
    for path in SOURCES:
        if sha((bundle / "sources" / path.relative_to(ROOT)).read_bytes()) != manifest["sources"][str(path.relative_to(ROOT))]:
            raise ValueError(f"saved source hash mismatch: {path.name}")
    profile = validate_profile(deepcopy(manifest["profile"]))
    expected = [f"{f['name']}__{r}" for f in profile["formats"] for r in profile["rules"]]
    if manifest["entries"] != expected:
        raise ValueError("manifest entries do not match the frozen profile")
    rows, accepted = [], []
    for fmt in profile["formats"]:
        for rule in profile["rules"]:
            name = f"{fmt['name']}__{rule}"
            directory = bundle / name
            if not (directory / "record.json").exists():
                rows.append({"rule_id": rule, "format": fmt["name"], "decision": "NOT_EVALUATED", "reason": "missing record"})
                continue
            record = read_json(directory / "record.json")
            if record.get("rule_id") != rule or record.get("format") != fmt["name"]:
                raise ValueError(f"record identity mismatch: {name}")
            if record["state"] != "COMPLETE":
                # Unfinished/error/domain-event labels are never imported as accepted rules.
                rows.append({"rule_id": rule, "format": fmt["name"], "decision": "NOT_EVALUATED",
                             "state": record["state"], "reported_decision": record["decision"],
                             "sampling": record.get("sampling", {}),
                             "reason": record.get("reason", "incomplete protocol")})
                continue
            if supplemental.unsupported(rule, fmt):
                raise ValueError(f"unsupported precision reported COMPLETE: {name}")
            lowerings = record["lowerings"]
            expected_sides = {"reference", "candidate", "oracle"} if fmt["compute"] == "fp64" else {"reference", "candidate"}
            if set(lowerings) != expected_sides:
                raise ValueError("missing compiled side")
            for side, digests in lowerings.items():
                if not digests or any(sha((directory / f"{side}.{i}.ptx").read_bytes()) != digest for i, digest in enumerate(digests)):
                    raise ValueError(f"compiled lowering hash mismatch: {name}/{side}")
            config = contract_for(profile, fmt, rule, manifest["backend"], manifest["sources"], lowerings)
            if config != record["config"] or supplemental.instance_key(config) != record["instance_key"]:
                raise ValueError(f"configuration identity mismatch: {name}")
            path = directory / "observations.npz"
            if sha(path.read_bytes()) != record["observations_sha256"]:
                raise ValueError(f"observation hash mismatch: {name}")
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            count = record["result"].get("completed_replicates")
            if type(count) is not int or count < 2:
                raise ValueError(f"invalid completed replicate count: {name}")
            sampling = domains.unpack(arrays, record, profile, count)
            if (set(arrays) != gates.OBSERVATIONS or any(a.dtype != np.float64 for a in arrays.values())
                    or arrays["delta"].shape != (count, 1)
                    or any(arrays[k].shape != (count,) for k in ("reference_error", "candidate_error"))):
                raise ValueError(f"observation shape/dtype mismatch: {name}")
            stop = gates.validate_stopping(arrays, profile["gates"]["vars"], profile["replicates"],
                                           profile["replicates_max"], profile["batch"])
            result = gates.evaluate(arrays, profile["gates"], seed_for(profile, fmt, rule), count, manifest["smoke"])
            result.update(stopping_reason=stop, completed_replicates=count)
            if result != record["result"] or result["decision"] != record["decision"]:
                raise ValueError(f"stored decision/statistics disagree with replay: {name}")
            row = {"rule_id": rule, "format": fmt["name"], "instance_key": record["instance_key"],
                   "decision": result["decision"], "bias": result["bias"]["status"], "vars": result["vars"]["status"],
                   "replicates": count, "stopping_reason": stop, "sampling": sampling,
                   "empirical_fallback": result["vars"]["empirical_fallback"],
                   "statistics": result, "relation": config["relation"]}
            rows.append(row)
            if result["decision"] in ACCEPTED:
                accepted.append({**row, "config": config, "artifact": str((directory / "record.json").resolve()),
                                 "observations_sha256": record["observations_sha256"]})
    return {"schema_version": 1, "supplement_schema": 1, "checker_version": gates.VERSION, "bundle": str(bundle.resolve()),
            "manifest_sha256": sha((bundle / "manifest.json").read_bytes()), "rows": rows, "accepted": accepted,
            "trust": "CPU replay validates statistics and identity; GPU execution and oracle remain trusted. "
                     "This table does not prove EvidenceValidated or IEEE equality."}



def publish_report(bundle, report, output):
    # Keep reports separate from raw bundles and the original admission table.
    # This command checks existing observations; it executes no returned code.
    if __package__:
        from . import report_numerics as reporting
    else:
        import report_numerics as reporting
    from collections import Counter
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for r in report["rows"]:
        stats = r.get("statistics")
        var = stats["vars"] if stats else {}
        rows.append({
            "rule": r["rule_id"], "format": r["format"], "replicates": r.get("replicates"),
            "z": reporting.maximum_z(stats["bias"].get("abs_z")) if stats else None,
            "B": stats["bias"].get("upper") if stats else None,
            "tau": stats["bias"].get("tau") if stats else None,
            "U": var.get("upper"), "bias": r.get("bias"), "vars": r.get("vars"),
            "u_kind": ("empirical_max" if var["empirical_fallback"] else var["branch"]) if stats else None,
            "accept": (r["decision"] in ACCEPTED if stats or r.get("state") in
                       {"UNSUPPORTED", "NUMERIC_EVENT", "ERROR"} else None),
            "decision": r.get("reported_decision", r["decision"]),
            "state": r.get("state", "COMPLETE" if stats else "PENDING"), "replayed": stats is not None,
            "reason": r.get("reason", ""),
            "valid_samples": r.get("sampling", {}).get("valid_samples"),
            "skipped_samples": r.get("sampling", {}).get("skipped_samples"),
        })
    table = {"rows": rows, "total": len(rows), "states": dict(Counter(r["state"] for r in rows)),
             "accepted": len(report["accepted"]), "replayed": sum(r["replayed"] for r in rows),
             **reporting.DEFINITIONS}
    reporting.publish(output, table)
    write_json(output / "admission.json", report)
    write_json(output / "experiment.json", read_json(bundle / "manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="validate a Python configuration without importing GPU packages")
    check.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    run_parser = sub.add_parser("run", help="execute paired Triton kernels on an NVIDIA GPU")
    run_parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--rules", help="comma-separated rule IDs; default comes from profile")
    run_parser.add_argument("--formats", help="comma-separated format names; default comes from profile")
    run_parser.add_argument("--smoke", action="store_true", help="32x33, four replicates, NEVER admissible")
    run_parser.add_argument("--resume", action="store_true", help="continue an interrupted identical run")
    replay_parser = sub.add_parser("import", help="recompute both gates from a returned bundle, CPU only")
    replay_parser.add_argument("bundle", type=Path)
    replay_parser.add_argument("--output", type=Path, required=True)
    report_parser = sub.add_parser("report", help="replay a bundle and write a reviewable JSON/CSV/Markdown table")
    report_parser.add_argument("bundle", type=Path)
    report_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            profile = validate_profile(deepcopy(load_module(args.profile.resolve()).PROFILE))
            print(json.dumps(profile, indent=2))
            return 0
        if args.command == "run":
            return run(args)
        report = replay(args.bundle)
        if args.command == "report":
            publish_report(args.bundle, report, args.output_dir)
            print(f"{len(report['accepted'])} accepted; supplemental report saved to {args.output_dir}")
            return 0
        with args.output.open("x") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
        print(f"Replayed {len(report['rows'])} rows; {len(report['accepted'])} accepted. Saved {args.output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
