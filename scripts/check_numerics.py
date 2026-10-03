#!/usr/bin/env python3
"""Check atomic Triton expression relations, or replay their results on CPU."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

if __package__:
    from . import numerical_registry as registry, numerical_gates as gates
else:
    import numerical_registry as registry
    import numerical_gates as gates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "experiments/floating_point/config.py"
KERNELS = ROOT / "experiments/floating_point/kernels.py"
SOURCES = [Path(__file__).resolve(), Path(gates.__file__), Path(registry.__file__), KERNELS, registry.CATALOG]
ACCEPTED = {"ACCEPT", "ACCEPT_WITH_WARNING"}
BUNDLE_VERSION = 6


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(registry.canonical_json(data) + b"\n")
    temporary.replace(path)


def read_json(path):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    result = json.loads(path.read_text(), object_pairs_hook=no_duplicates)
    registry.canonical_json(result)
    return result


def load_module(path):
    spec = importlib.util.spec_from_file_location("veritile_numerics_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_profile(profile):
    expected = {"shape", "distribution", "seed", "replicates", "replicates_max", "batch",
                "formats", "rules", "launch", "gates"}
    if type(profile) is not dict or set(profile) != expected:
        raise ValueError(f"profile requires exactly {sorted(expected)}")
    registry.canonical_json(profile)
    shape = profile["shape"]
    if type(shape) is not list or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("shape must be [rows, columns] with positive concrete integers")
    dist = profile["distribution"]
    if (type(dist) is not dict or set(dist) != {"family", "mean", "std"}
            or dist["family"] != "normal" or type(dist["mean"]) not in (int, float)
            or type(dist["std"]) not in (int, float) or dist["std"] <= 0):
        raise ValueError("distribution must specify normal mean and positive std (sigma)")
    for key in ("seed", "replicates", "replicates_max", "batch"):
        if type(profile[key]) is not int or profile[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if profile["replicates"] < 2 or profile["seed"] >= 2**63:
        raise ValueError("need >= 2 replicates and seed < 2**63")
    if profile["replicates_max"] < profile["replicates"] or profile["batch"] < 2:
        raise ValueError("replicates_max must be >= replicates; batch must be >= 2")
    names = set()
    if type(profile["formats"]) is not list or not profile["formats"]:
        raise ValueError("formats must be a nonempty list")
    for fmt in profile["formats"]:
        if type(fmt) is not dict or set(fmt) != {"name", "input", "compute", "accumulator", "output"}:
            raise ValueError("each format requires name/input/compute/accumulator/output")
        if not isinstance(fmt["name"], str) or not re.fullmatch(r"[A-Za-z0-9_]+", fmt["name"]) or fmt["name"] in names:
            raise ValueError("format names must be unique alphanumeric/underscore identifiers")
        names.add(fmt["name"])
        if any(fmt[k] not in ("bf16", "fp32") for k in ("input", "compute", "output")):
            raise ValueError("input/compute/output formats must be bf16 or fp32")
        if fmt["accumulator"] != "fp32":
            raise ValueError("the local ACC-WIDEN relation uses fp32 for widened additions")
    if profile["rules"] == "all":
        profile["rules"] = list(registry.load_catalog())
    if (type(profile["rules"]) is not list or not profile["rules"]
            or any(type(r) is not str or r not in registry.load_catalog() for r in profile["rules"])
            or len(set(profile["rules"])) != len(profile["rules"])):
        raise ValueError("rules must be 'all' or distinct atomic rule IDs; composite transformations are not admitted")
    launch = profile["launch"]
    if type(launch) is not dict or set(launch) != {"block", "num_warps"}:
        raise ValueError("launch requires block/num_warps")
    block = launch["block"]
    if type(block) is not int or block < 32 or block > 65536 or block & (block - 1):
        raise ValueError("block must be a power of two in [32, 65536]")
    if launch["num_warps"] not in (4, 8):
        raise ValueError("num_warps must be 4/8")
    if ((shape[0] * shape[1] + block - 1) // block) * block > 2**31:
        raise ValueError("elementwise offsets must fit in signed 32-bit indices")
    g = profile["gates"]
    if type(g) is not dict or set(g) != {"bias", "vars", "warning_policy"}:
        raise ValueError("gates requires bias/vars/warning_policy")
    if (type(g["bias"]) is not dict or set(g["bias"]) != {"tau", "se_multiplier"}
            or type(g["vars"]) is not dict or set(g["vars"]) != {
                "quantile", "horizon", "alpha", "bootstrap", "min_exceedances", "warn", "fail"}):
        raise ValueError("unexpected two-gates parameters")
    if any(type(x) not in (int, float) or x <= 0 for c in (g["bias"], g["vars"]) for x in c.values()):
        raise ValueError("gate parameters must be positive finite numbers")
    v = g["vars"]
    if not 0 < v["alpha"] < 0.5 or not 0 < v["quantile"] < 1 or v["warn"] >= v["fail"] or v["horizon"] <= 1:
        raise ValueError("invalid quantile, horizon or ordered vars thresholds")
    if (type(v["bootstrap"]) is not int or v["bootstrap"] < 2
            or type(v["min_exceedances"]) is not int or v["min_exceedances"] < 3):
        raise ValueError("bootstrap >= 2 and min_exceedances >= 3 must be integers")
    if g["warning_policy"] not in ("pass_only", "allow_warn"):
        raise ValueError("warning_policy must be pass_only or allow_warn")
    return profile


def source_hashes():
    return {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in SOURCES}


def seed_for(profile, fmt, rule):
    data = registry.canonical_json([profile["seed"], fmt["name"], rule])
    return int(sha(data)[:15], 16)


def shapes_for(profile, rule):
    # Shape describes a batch of independent local expressions, not a reduction.
    return {name: list(profile["shape"]) for name in ("a", "b", "c", "out")}


def graph_hash(rule, side, profile, fmt, sources):
    # Bind the host sampler/oracle and importer too, not just the JIT function.
    return sha(registry.canonical_json({"implementation_sources": sources,
               "rule": rule, "side": side, "shape": profile["shape"], "formats": fmt,
               "launch": profile["launch"]}))


def contract_for(profile, fmt, rule, backend, sources, lowerings):
    shapes = shapes_for(profile, rule)
    return {
        "schema_version": 1, "rule_id": rule,
        **{side: {"graph_sha256": graph_hash(rule, side, profile, fmt, sources),
                  "lowering_sha256": sha(registry.canonical_json(lowerings[side]))}
           for side in ("reference", "candidate")},
        "numerics": {
            "semantics_version": "triton-atomic-relations", "input_formats": {k: fmt["input"] for k in ("a", "b", "c")},
            "node_formats": {"elementwise": fmt["compute"],
                             "explicit_cast_target": "bf16", "fma_intrinsic": "fp32",
                             "template_details": "see bound Triton source for every cast and operation"},
            "accumulator_formats": {"reference": fmt["input"], "candidate": fmt["accumulator"]}
                                   if rule == "ACC-WIDEN" else {},
            "output_formats": {"out": fmt["output"]}, "rounding": "rne casts; instruction-specific intrinsics",
            "nan": "no NaN payload equivalence; nonfinite errors produce K=inf and FAIL",
            "subnormal": "native bound GPU/compiler instruction behavior; no software flush substitution",
            "intrinsics": {"fma": "tl.fma", "sqrt": "tl.sqrt", "rsqrt": "tl.rsqrt", "div": "tl.div_rn",
                           "oracle": "torch fp64 on the same quantized input"}},
        "layout": {"shapes": shapes, "strides": {k: [v[1], 1] if len(v) == 2 else [1] for k, v in shapes.items()},
                   "reduction": None, "scan": None, "dot": None},
        "backend": backend,
        "probe": {"family": "normal", "roles": {k: profile["distribution"] for k in ("a", "b", "c")},
                  "joint_distribution": "independent roles and elements; fresh whole tuple per replicate",
                  "weights": None, "seed": seed_for(profile, fmt, rule),
                  "quantization": "torch fp64 normal -> input dtype (RNE); oracle widens those same values",
                  "special_values": "no truncation/resampling; domain/nonfinite events recorded"},
        "protocol": {"name": "two-gates", "version": gates.VERSION, "checker_version": sources["scripts/numerical_gates.py"],
                     "bias": {**profile["gates"]["bias"], "replicates": profile["replicates"],
                              "buckets": "one mean per replicate across all IID scalar instances; no positional buckets",
                              "ulp": "per-element output-format ULP at abs(golden) rounded to output dtype; inward at max finite; minimum subnormal spacing at zero",
                              "delta": "mean over all elements of (candidate-reference)/local_golden_ulp; normalize before averaging",
                              "acceptance": "abs(mean) + se_multiplier * std / sqrt(R) <= tau across replicate means; z is diagnostic",
                              "coverage": "engineering SE bands; no calibrated simultaneous or optional-stopping coverage"},
                     "vars": {**profile["gates"]["vars"], "replicates": profile["replicates"],
                              "errors": "separate maxima of absolute oracle errors for reference and candidate; no ULP normalization",
                              "ratio": "K=Ec/Er; both zero gives 0; Er=0<Ec gives infinity; no additive allowance",
                              "tail": "PWM, xi clipped <= 0, seed-0 bootstrap, empirical-max fallback",
                              "replicates_max": profile["replicates_max"], "batch": profile["batch"],
                              "stopping": "full batches; min replicates then magnitude band stable or empirical fallback; nonfinite stops immediately"},
                     "decision_policy": profile["gates"]["warning_policy"]},
    }


class NumericEvent(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def oracle(torch, rule, inputs):
    a, b, c = (x.double() for x in inputs)
    if rule in ("ADD-COMMUTE", "CAST-MOVE"):
        return a + b
    if rule == "MUL-COMMUTE":
        return a * b
    if rule in ("ADD-ASSOC", "ACC-WIDEN"):
        return (a + b) + c
    if rule == "MUL-ASSOC":
        return (a * b) * c
    if rule == "MUL-DISTRIB":
        return a * (b + c)
    if rule == "FMA-CONTRACT":
        return a * b + c
    if rule == "CAST-REMOVE":
        return (a + b) * c
    if rule == "DIV-RCP":
        if bool((b == 0).any()):
            raise NumericEvent("INCONCLUSIVE", "division domain violated by a sampled zero denominator")
        return a / b
    if rule == "SQRT-RSQRT":
        if bool((a <= 0).any()):
            raise NumericEvent("INCONCLUSIVE", "sqrt domain violated; the requested normal distribution was not conditioned")
        return torch.rsqrt(a)
    if rule in ("ROUND-IDEM", "BF16-WIDEN-RETURN", "CANCEL"):
        return a
    raise ValueError("no oracle for atomic rule " + rule)


def launch_pair(torch, triton, kernels, rule, inputs, profile, fmt):
    if rule not in kernels.SUPPORTED:
        raise ValueError("no atomic template for " + rule)
    m, n = profile["shape"]
    count = m * n
    a, b, c = inputs
    options = {"num_warps": profile["launch"]["num_warps"], "enable_fp_fusion": False}
    block = profile["launch"]["block"]
    out_shape = shapes_for(profile, rule)["out"]
    dtype = torch.bfloat16 if fmt["output"] == "bf16" else torch.float32
    outputs, compiled = [], {}
    for side, name in enumerate(("reference", "candidate")):
        out = torch.empty(out_shape, dtype=dtype, device=a.device)
        programs = [kernels.elementwise[(triton.cdiv(count, block),)](
            a, b, c, out, count, rule, side, fmt["compute"] == "bf16", fmt["input"] == "bf16", block, **options)]
        outputs.append(out)
        compiled[name] = [p.asm["ptx"] for p in programs]
    return outputs, compiled


def ulp(torch, magnitude, dtype):
    """Working-format ULP, including rounding and the max-finite inward gap."""
    comparison_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    scale = magnitude.detach().abs().to(comparison_dtype)
    upper = torch.nextafter(scale, torch.full_like(scale, float("inf")))
    lower = torch.nextafter(scale, torch.zeros_like(scale))
    # Only max-finite values may use the inward gap. An overflowed golden cast
    # must yield NaN, not an infinite scale that would normalize every error to 0.
    return torch.where(torch.isfinite(scale) & torch.isinf(upper),
                       scale.double() - lower.double(), upper.double() - scale.double())


def observe(torch, reference, candidate, exact, output_format, errors=None):
    """Keep a local-ULP bias mean and absolute oracle-error peaks per replicate.

    Positions in these atomic probes share a distribution and expression. They
    are not distinct channels. Retain a singleton bucket dimension for the
    shared gate API; R still counts independent whole-input replicates.

    Optional elementwise oracle errors preserve residual-based evaluations when
    subtracting a rounded golden value would lose the true rounding residual.
    """
    # Keep invalid observations: both gates reject them, and replay sees them too.
    ref, cand = reference.double(), candidate.double()
    scale = ulp(torch, exact, output_format)
    delta = (cand - ref) / scale
    if errors is None:
        errors = ((ref - exact).abs(), (cand - exact).abs())
    if len(errors) != 2 or any(error.shape != exact.shape for error in errors):
        raise ValueError("oracle errors must match the golden tensor shape")
    return {
        "delta": delta.mean().reshape(1).cpu().numpy(),
        "reference_error": errors[0].max().item(),
        "candidate_error": errors[1].max().item(),
    }


def run_instance(torch, triton, kernels, profile, fmt, rule, directory, backend, sources, smoke):
    directory.mkdir(exist_ok=True)
    record_path = directory / "record.json"
    base = {"rule_id": rule, "format": fmt["name"], "state": "RUNNING", "decision": "NOT_EVALUATED"}
    write_json(record_path, base)
    if rule == "BF16-WIDEN-RETURN" and (fmt["input"] != "bf16" or fmt["output"] != "bf16"):
        write_json(record_path, {**base, "state": "UNSUPPORTED", "reason": "requires bf16 input and output"})
        return "UNSUPPORTED"
    observations = {key: [] for key in gates.OBSERVATIONS}
    seed = seed_for(profile, fmt, rule)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    dtype = torch.bfloat16 if fmt["input"] == "bf16" else torch.float32
    lowerings = None
    started = time.monotonic()
    try:
        limit = ((profile["replicates_max"] + profile["batch"] - 1) // profile["batch"]) * profile["batch"]
        for replicate in range(limit):
            dist = profile["distribution"]
            shapes = shapes_for(profile, rule)
            inputs = [(torch.randn(shapes[key], device="cuda", dtype=torch.float64, generator=generator)
                       * dist["std"] + dist["mean"]).to(dtype) for key in ("a", "b", "c")]
            if any(not bool(torch.isfinite(x).all()) for x in inputs):
                raise NumericEvent("INCONCLUSIVE", "input quantization produced nonfinite values")
            exact = oracle(torch, rule, inputs)
            (reference, candidate), compiled = launch_pair(torch, triton, kernels, rule, inputs, profile, fmt)
            if lowerings is None:
                lowerings = {}
                for side, programs in compiled.items():
                    lowerings[side] = []
                    for index, ptx in enumerate(programs):
                        name = f"{side}.{index}.ptx"
                        (directory / name).write_text(ptx)
                        lowerings[side].append(sha(ptx.encode()))
                config = contract_for(profile, fmt, rule, backend, sources, lowerings)
                base.update(config=config, instance_key=registry.instance_key(config), lowerings=lowerings)
                write_json(record_path, base)
            elif any([sha(p.encode()) for p in compiled[side]] != lowerings[side] for side in lowerings):
                raise ValueError("compiled lowering changed within one instance")
            sample = observe(torch, reference, candidate, exact, fmt["output"])
            for key in observations:
                observations[key].append(sample[key])
            if replicate == 0 or (replicate + 1) % 32 == 0:
                print(f"  {fmt['name']} {rule}: {replicate + 1}/{limit} ({time.monotonic() - started:.1f}s)", flush=True)
            if (replicate + 1) % profile["batch"] == 0:
                var, stop = gates.checkpoint(observations, profile["gates"]["vars"],
                                             profile["replicates"], profile["replicates_max"], profile["batch"])
                print(f"  magnitude: {var['status']} U={var['upper']} stop={stop}", flush=True)
                if stop:
                    break
        arrays = {key: np.asarray(values, dtype=np.float64) for key, values in observations.items()}
        count = len(arrays["delta"])
        result = gates.evaluate(arrays, profile["gates"], seed, count, smoke)
        result.update(stopping_reason=stop, completed_replicates=count)
        np.savez_compressed(directory / "observations.npz", **arrays)
        base.update(state="COMPLETE", result=result, decision=result["decision"],
                    observations_sha256=sha((directory / "observations.npz").read_bytes()),
                    seconds=time.monotonic() - started)
    except NumericEvent as event:
        base.update(state="NUMERIC_EVENT", decision="SMOKE_ONLY" if smoke else event.status,
                    reason=str(event), completed_replicates=len(observations["delta"]))
    except Exception as error:
        base.update(state="ERROR", decision="NOT_EVALUATED", reason=f"{type(error).__name__}: {error}",
                    completed_replicates=len(observations["delta"]))
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
        raise ValueError("GPU run requires torch and triton in your CUDA environment; see experiments/floating_point/README.md") from error
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise ValueError("this runner requires an NVIDIA CUDA GPU; no CPU numerical fallback is used")
    kernels = load_module(KERNELS)
    if kernels.SUPPORTED != set(registry.load_catalog()):
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
            (source_dir / path.name).write_bytes(path.read_bytes())
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
        if sha((bundle / "sources" / path.name).read_bytes()) != manifest["sources"][str(path.relative_to(ROOT))]:
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
                             "reason": record.get("reason", "incomplete protocol")})
                continue
            lowerings = record["lowerings"]
            if set(lowerings) != {"reference", "candidate"}:
                raise ValueError("missing compiled side")
            for side, digests in lowerings.items():
                if not digests or any(sha((directory / f"{side}.{i}.ptx").read_bytes()) != digest for i, digest in enumerate(digests)):
                    raise ValueError(f"compiled lowering hash mismatch: {name}/{side}")
            config = contract_for(profile, fmt, rule, manifest["backend"], manifest["sources"], lowerings)
            if config != record["config"] or registry.instance_key(config) != record["instance_key"]:
                raise ValueError(f"configuration identity mismatch: {name}")
            path = directory / "observations.npz"
            if sha(path.read_bytes()) != record["observations_sha256"]:
                raise ValueError(f"observation hash mismatch: {name}")
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            count = record["result"].get("completed_replicates")
            if type(count) is not int or count < 2:
                raise ValueError(f"invalid completed replicate count: {name}")
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
                   "replicates": count, "stopping_reason": stop,
                   "empirical_fallback": result["vars"]["empirical_fallback"]}
            rows.append(row)
            if result["decision"] in ACCEPTED:
                accepted.append({**row, "config": config, "artifact": str((directory / "record.json").resolve()),
                                 "observations_sha256": record["observations_sha256"]})
    return {"schema_version": 1, "checker_version": gates.VERSION, "bundle": str(bundle.resolve()),
            "manifest_sha256": sha((bundle / "manifest.json").read_bytes()), "rows": rows, "accepted": accepted,
            "trust": "CPU replay validates statistics and identity; GPU execution and oracle remain trusted. "
                     "This table does not prove EvidenceValidated or IEEE equality."}


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
    args = parser.parse_args()
    try:
        if args.command == "check":
            profile = validate_profile(deepcopy(load_module(args.profile.resolve()).PROFILE))
            print(json.dumps(profile, indent=2))
            return 0
        if args.command == "run":
            return run(args)
        report = replay(args.bundle)
        with args.output.open("x") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
        print(f"Replayed {len(report['rows'])} rows; {len(report['accepted'])} accepted. Saved {args.output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
