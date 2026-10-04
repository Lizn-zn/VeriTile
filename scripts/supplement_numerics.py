"""Supplemental catalogue, contracts and oracles using the shared two gates.

No GPU dependency at import time. This module reuses the original profile
validator and observation definitions, without modifying their module globals.
"""
from copy import deepcopy
from pathlib import Path

if __package__:
    from . import check_numerics as original
else:
    import check_numerics as original

ROOT = original.ROOT
DIRECTORY = ROOT / "experiments/floating_point/supplement"
CATALOG = DIRECTORY / "rules.json"
DEFAULT_PROFILE = DIRECTORY / "config.py"
KERNELS = DIRECTORY / "kernels.py"
NumericEvent = original.NumericEvent
COUNT_RULES = {"COUNT-ZERO", "COUNT-SUCCESSOR"}
PAIRED_INPUTS = {"LOG-MUL-LIBDEVICE": "LOG-MUL", "LOG-EXP-FULL-LIBDEVICE": "LOG-EXP-LIBDEVICE",
                 "LOG-EXP-EXPM1": "LOG-EXP-LIBDEVICE"}


def seed_for(profile, fmt, rule):
    """Library comparisons share input draws while retaining distinct contracts."""
    return original.seed_for(profile, fmt, PAIRED_INPUTS.get(rule, rule))


def load_catalog():
    data = original.read_json(CATALOG)
    rules = data["rules"]
    result = {r["id"]: r for r in rules}
    if data["schema_version"] != 1 or len(result) != len(rules):
        raise ValueError("invalid supplemental catalogue")
    return result


def validate_profile(profile):
    # Validate shape/distribution/gates using the shared checker.
    # Only the catalogue and explicitly supported precision tuples differ.
    if type(profile) is not dict or type(profile.get("formats")) is not list:
        raise ValueError("profile must be an object with a formats list")
    projection = deepcopy(profile)
    projection["rules"] = ["ADD-COMMUTE"]
    integer = profile.get("distribution", {}).get("family") == "uniform_integer"
    if integer:
        dist = profile["distribution"]
        if (set(dist) != {"family", "low", "high"}
                or any(type(dist[k]) is not int for k in ("low", "high"))
                or not 0 <= dist["low"] < dist["high"] <= 2**31 - 1):
            raise ValueError("integer counts require 0 <= low < high <= 2**31-1")
        projection["distribution"] = {"family": "normal", "mean": 1.0, "std": 1.0}
    for fmt in projection["formats"]:
        if type(fmt) is not dict:
            raise ValueError("each precision profile must be an object")
        if fmt.get("input") == "int32":
            if not integer or any(fmt.get(k) != "fp32" for k in ("compute", "accumulator", "output")):
                raise ValueError("integer counts require int32 input and fp32 arithmetic/output")
            fmt["input"] = "fp32"
        elif integer:
            raise ValueError("integer-count distribution requires int32 input")
        elif fmt.get("compute") == "fp64":
            if (fmt.get("input"), fmt.get("accumulator"), fmt.get("output")) != ("fp64", "fp64", "fp32"):
                raise ValueError("fp64 work requires fp64 operands/accumulator and fp32 output")
            fmt["input"] = fmt["compute"] = fmt["accumulator"] = "fp32"
        elif (fmt.get("input"), fmt.get("compute"), fmt.get("output")) not in {
            ("bf16", "bf16", "bf16"), ("bf16", "fp32", "bf16"), ("fp32", "fp32", "fp32")
        }:
            raise ValueError("unsupported supplemental precision tuple")
    original.validate_profile(projection)
    if profile["rules"] == "all":
        profile["rules"] = list(load_catalog())
    rules = profile["rules"]
    if (type(rules) is not list or not rules
            or any(type(r) is not str or r not in load_catalog() for r in rules)
            or len(set(rules)) != len(rules)):
        raise ValueError("rules must be 'all' or distinct supplemental atomic rule IDs")
    if integer and not set(rules) <= COUNT_RULES:
        raise ValueError("integer-count profiles support only count-conversion relations")
    original.registry.canonical_json(profile)
    return profile


def unsupported(rule, fmt):
    if rule in COUNT_RULES:
        if (fmt["input"], fmt["compute"], fmt["output"]) != ("int32", "fp32", "fp32"):
            return "count conversion requires int32 inputs and fp32 arithmetic/output"
        return None
    if fmt["input"] == "int32":
        return "int32 inputs apply only to count conversion"
    if rule in {"EXP-SUB-INTRINSIC", "LOG-EXP-EXPM1"} and (fmt["input"], fmt["compute"], fmt["output"]) != ("fp32", "fp32", "fp32"):
        return "this experiment covers only fp32 inputs, arithmetic and outputs"
    if fmt["compute"] == "fp64" and rule != "DIV-MUL-RCP":
        return "fp64-work supplement covers only ordinary division with fp32 output"
    return None


def instance_key(config):
    if config["rule_id"] not in load_catalog() or config.get("supplement_schema") != 1:
        raise ValueError("unknown supplemental atomic contract")
    return original.sha(b"veritile.numerical.supplement\0" + original.registry.canonical_json(config))


def contract_for(profile, fmt, rule, backend, sources, lowerings):
    config = original.contract_for(profile, fmt, rule, backend, sources, lowerings)
    config.update(supplement_schema=1, relation=deepcopy(load_catalog()[rule]))
    config["relation"]["final_cast"] = fmt["output"]
    config["numerics"].update(
        semantics_version="triton-supplemental-scalar-relations-1",
        node_formats={"arithmetic": fmt["compute"], "transcendental": fmt["compute"],
                      "details": "bf16 nodes execute in fp32 then explicitly round bf16; see bound source"},
        accumulator_formats={},  # Every expression is scalar; no reduction accumulator.
        intrinsics={"div": "ordinary Triton /", "exp": "libdevice.exp" if rule in {"EXP-SUB", "LOG-EXP-LIBDEVICE", "LOG-EXP-FULL-LIBDEVICE", "LOG-EXP-EXPM1"} else "tl.exp",
                    "log": "libdevice.log" if rule in PAIRED_INPUTS else "tl.log", "max": "tl.maximum",
                    "oracle": "torch fp64 mathematical reference on the same quantized operands"})
    config["probe"]["special_values"]["literals"] = "only explicit -inf literals in the relation"
    config["probe"]["active_operands"] = load_catalog()[rule]["operands"]
    if rule in PAIRED_INPUTS:
        config["probe"].update(seed=seed_for(profile, fmt, rule), paired_input_rule=PAIRED_INPUTS[rule])
    if rule == "LOG-EXP-EXPM1":
        config["numerics"]["intrinsics"].update(expm1="libdevice.expm1", log1p="libdevice.log1p")
        config["relation"]["branch"] = {
            "condition": "abs(a) <= 0.5", "threshold": 0.5,
            "inside": "fp32 log1p(fp32 expm1(a))",
            "outside": "fp32 log(fp32 exp(a))",
            "inactive_arguments": "zero before evaluating unused tl.where arms",
        }
    if fmt["compute"] == "fp64":
        config["numerics"]["intrinsics"]["oracle"] = (
            "quotient error |fma(-output,b,a)/b|; explicit fp64 tl.fma followed by fp64 division; "
            "avoids subtracting an already rounded fp64 quotient")
        config["numerics"]["intrinsics"]["oracle_lowering_sha256"] = original.sha(
            original.registry.canonical_json(lowerings["oracle"]))
        config["relation"]["scope"] = "fp64 arithmetic INSIDE the final fp32 cast; not bare fp64 equality"
        config["probe"]["quantization"] = "torch fp64 normal -> fp64 local operands; no intervening fp32 quantization"
    if rule in COUNT_RULES:
        dist = profile["distribution"]
        config["probe"].update(
            family="constant" if rule == "COUNT-ZERO" else "uniform_integer",
            roles={} if rule == "COUNT-ZERO" else {"a": deepcopy(dist)},
            joint_distribution=("deterministic constant; repeats do not extend the tested domain"
                                if rule == "COUNT-ZERO" else
                                "independent uniform integer elements; fresh counts per replicate"),
            quantization="exact int32 counts; conversion to fp32 occurs inside the tested kernel")
        config["numerics"]["intrinsics"] = {
            "conversion": "int32 to fp32, round to nearest even",
            "add": "reference int32 addition; candidate fp32 addition",
            "oracle": "exact integer count represented in torch fp64"}
        config["relation"]["domain"] = ("constant integer zero" if rule == "COUNT-ZERO"
            else f"integer a; {dist['low']} <= a < {dist['high']}; int32 a+1 cannot overflow")
    return config


def sample_inputs(torch, profile, fmt, rule, generator):
    if rule in COUNT_RULES:
        dist = profile["distribution"]
        if rule == "COUNT-ZERO":
            a = torch.zeros(profile["shape"], device="cuda", dtype=torch.int32)
        else:
            a = torch.randint(dist["low"], dist["high"], profile["shape"],
                              device="cuda", dtype=torch.int32, generator=generator)
        return [a, a, a]  # Only a is used; zero conversion has no variable operand.
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}[fmt["input"]]
    dist = profile["distribution"]
    return [(torch.randn(profile["shape"], device="cuda", dtype=torch.float64, generator=generator)
             * dist["std"] + dist["mean"]).to(dtype) for _ in range(3)]


def oracle(torch, rule, inputs):
    a, b, c = (x.double() for x in inputs)
    if rule == "DIV-MUL-RCP":
        return a / b
    if rule == "MUL-RCP-CANCEL":
        return torch.ones_like(a)
    if rule in {"LOG-MUL", "LOG-MUL-LIBDEVICE"}:
        return torch.log(a * b)
    if rule in {"EXP-SUB", "EXP-SUB-INTRINSIC"}:
        return torch.exp(a - b)
    if rule == "COUNT-ZERO":
        return torch.zeros_like(a)
    if rule == "COUNT-SUCCESSOR":
        return a + 1
    if rule == "EXP-ZERO":
        return torch.ones_like(a)
    if rule == "EXP-NEG-INF-SUB":
        return torch.zeros_like(a)
    if rule in {"MAX-COMMUTE", "MAX-ASSOC"}:
        ab = torch.maximum(a, b)
        return torch.maximum(ab, c) if rule == "MAX-ASSOC" else ab
    if rule in {"ADD-ZERO", "MUL-ONE", "DIV-ONE", "LOG-EXP", "LOG-EXP-LIBDEVICE", "LOG-EXP-FULL-LIBDEVICE", "LOG-EXP-EXPM1", "MAX-IDEM", "MAX-NEG-INF"}:
        return a
    raise ValueError("unknown supplemental oracle")


def observe(torch, reference, candidate, exact, fmt, errors=None, valid=None):
    if fmt["compute"] == "fp64":
        if errors is None or len(errors) != 2:
            raise ValueError("fp64-work instance requires the bound residual oracle")
    return original.observe(torch, reference, candidate, exact, fmt["output"], errors, valid)


def launch_pair(torch, triton, kernels, rule, inputs, profile, fmt):
    if unsupported(rule, fmt):
        raise ValueError(unsupported(rule, fmt))
    count = inputs[0].numel()
    block = profile["launch"]["block"]
    dtype = torch.bfloat16 if fmt["output"] == "bf16" else torch.float32
    outputs, compiled = [], {}
    for side, name in enumerate(("reference", "candidate")):
        out = torch.empty_like(inputs[0], dtype=dtype)
        program = kernels.elementwise[(triton.cdiv(count, block),)](
            *inputs, out, count, rule, side, fmt["compute"], block,
            num_warps=profile["launch"]["num_warps"], enable_fp_fusion=False)
        outputs.append(out)
        compiled[name] = [program.asm["ptx"]]
    errors = None
    if fmt["compute"] == "fp64":
        errors = [torch.empty_like(inputs[0], dtype=torch.float64) for _ in range(2)]
        program = kernels.quotient_errors[(triton.cdiv(count, block),)](
            inputs[0], inputs[1], *outputs, *errors, count, block,
            num_warps=profile["launch"]["num_warps"], enable_fp_fusion=False)
        compiled["oracle"] = [program.asm["ptx"]]
    return outputs, compiled, errors
