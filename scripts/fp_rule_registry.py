"""Register floating-point rule instances; never infer acceptance from a rule ID.

This module validates identity dimensions and initializes pending records. It
does not execute graphs, verify backend conformance, or manufacture gate/proof
evidence. The gate runner will populate records after checking those contracts.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "experiments/floating_point/rules.json"
SCHEMA_VERSION = 1

# Every dimension is mandatory, including an explicit null for inapplicable
# plans. Per-operation formats are part of graph identity AND this run contract.
DIMENSIONS = {
    "reference": {"graph_sha256", "lowering_sha256"},
    "candidate": {"graph_sha256", "lowering_sha256"},
    "numerics": {"semantics_version", "input_formats", "node_formats", "accumulator_formats",
                 "output_formats", "rounding", "nan", "subnormal", "intrinsics"},
    "layout": {"shapes", "strides", "reduction", "scan", "dot"},
    "backend": {"kind", "implementation_version", "target", "compiler", "compile_options", "launch"},
    "probe": {"family", "roles", "joint_distribution", "weights", "seed", "quantization", "special_values"},
    "protocol": {"name", "version", "checker_version", "bias", "vars", "decision_policy"},
}


def _check_json(value):
    """Disallow implicit coercions (e.g. int keys or tuples) before hashing."""
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("configuration object keys must be strings")
        for child in value.values():
            _check_json(child)
    elif type(value) is list:
        for child in value:
            _check_json(child)
    elif value is not None and type(value) not in (str, bool, int, float):
        raise ValueError("configuration must contain JSON values only")


def canonical_json(value):
    _check_json(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def load_catalog(path=CATALOG):
    data = json.loads(Path(path).read_text())
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported catalog schema")
    rules = data["rules"]
    ids = [rule["id"] for rule in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate rule ID")
    if any(rule["category"] not in ("exact_candidate", "two_gates_candidate") for rule in rules):
        raise ValueError("unknown rule category")
    return {rule["id"]: rule for rule in rules}


def validate_config(config):
    if type(config) is not dict:
        raise ValueError("configuration must be an object")
    expected = {"schema_version", "rule_id", *DIMENSIONS}
    if set(config) != expected or type(config["schema_version"]) is not int or config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"configuration requires exactly these fields: {sorted(expected)}")
    if not isinstance(config["rule_id"], str) or config["rule_id"] not in load_catalog():
        raise ValueError("unknown rule ID")
    for name, fields in DIMENSIONS.items():
        if type(config[name]) is not dict or set(config[name]) != fields:
            raise ValueError(f"{name} requires exactly these fields: {sorted(fields)}")
        nullable = {("layout", "reduction"), ("layout", "scan"), ("layout", "dot"),
                    ("probe", "weights"), ("backend", "compiler")}
        for field, value in config[name].items():
            if value is None and (name, field) not in nullable:
                raise ValueError(f"{name}.{field} must be explicitly specified")
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{name}.{field} cannot be empty")
    for side in ("reference", "candidate"):
        for digest in config[side].values():
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{side} requires lowercase SHA-256 identities")
    shapes, strides = config["layout"]["shapes"], config["layout"]["strides"]
    if type(shapes) is not dict or not shapes or type(strides) is not dict or shapes.keys() != strides.keys():
        raise ValueError("shapes and strides must describe the same named tensors")
    for name, shape in shapes.items():
        stride = strides[name]
        if type(shape) is not list or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("each shape must be a list of concrete nonnegative integers")
        if type(stride) is not list or len(stride) != len(shape) or any(type(n) is not int for n in stride):
            raise ValueError("each stride must be an integer list matching its shape")
    for field in ("input_formats", "node_formats", "output_formats"):
        if type(config["numerics"][field]) is not dict or not config["numerics"][field]:
            raise ValueError(f"numerics.{field} must identify formats by name")
    canonical_json(config)  # Also rejects NaN and infinity anywhere in a config.


def instance_key(config):
    validate_config(config)
    return hashlib.sha256(b"veritile.fp.instance.v1\0" + canonical_json(config)).hexdigest()


def pending_record(config):
    """Both exact-proof candidates and statistical candidates start unaccepted."""
    key = instance_key(config)
    # Detach from caller-owned mutable dictionaries.
    snapshot = json.loads(canonical_json(config))
    return {
        "schema_version": SCHEMA_VERSION,
        "instance_key": key,
        "config": snapshot,
        "strict_fp": {"status": "UNPROVEN", "evidence": None},
        "bias": {"status": "NOT_RUN", "evidence": None},
        "vars": {"status": "NOT_RUN", "evidence": None},
        "decision": "NOT_EVALUATED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list candidate rules, without acceptance claims")
    init = sub.add_parser("init", help="register a fully identified configuration as NOT_EVALUATED")
    init.add_argument("config", type=Path)
    init.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "list":
            print(json.dumps(list(load_catalog().values()), indent=2))
        else:
            record = pending_record(json.loads(args.config.read_text()))
            # A previous run's evidence must not be silently overwritten.
            with args.output.open("x") as output:
                json.dump(record, output, indent=2, allow_nan=False)
                output.write("\n")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
