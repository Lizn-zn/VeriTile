"""Supplemental catalogue, contracts and oracles; the original run is frozen.

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


def load_catalog():
    data = original.read_json(CATALOG)
    rules = data["rules"]
    result = {r["id"]: r for r in rules}
    if data["schema_version"] != 1 or len(result) != len(rules):
        raise ValueError("invalid supplemental catalogue")
    return result


def validate_profile(profile):
    # Validate the unchanged shape/distribution/gates using the frozen checker.
    # Only the catalogue and explicitly supported precision tuples differ.
    if type(profile) is not dict or type(profile.get("formats")) is not list:
        raise ValueError("profile must be an object with a formats list")
    projection = deepcopy(profile)
    projection["rules"] = ["ADD-COMMUTE"]
    for fmt in projection["formats"]:
        if type(fmt) is not dict:
            raise ValueError("each precision profile must be an object")
        if fmt.get("compute") == "fp64":
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
    original.registry.canonical_json(profile)
    return profile


def unsupported(rule, fmt):
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
        intrinsics={"div": "ordinary Triton /", "exp": "tl.exp", "log": "tl.log", "max": "tl.maximum",
                    "oracle": "torch fp64 mathematical reference on the same quantized operands"})
    config["probe"]["special_values"] = "only explicit -inf literals in the relation; no conditioning or resampling"
    config["probe"]["active_operands"] = load_catalog()[rule]["operands"]
    if fmt["compute"] == "fp64":
        config["numerics"]["intrinsics"]["oracle"] = (
            "quotient error |fma(-output,b,a)/b|; explicit fp64 tl.fma followed by fp64 division; "
            "avoids subtracting an already rounded fp64 quotient")
        config["numerics"]["intrinsics"]["oracle_lowering_sha256"] = original.sha(
            original.registry.canonical_json(lowerings["oracle"]))
        config["relation"]["scope"] = "fp64 arithmetic INSIDE the final fp32 cast; not bare fp64 equality"
        config["probe"]["quantization"] = "torch fp64 normal -> fp64 local operands; no intervening fp32 quantization"
    return config


def oracle(torch, rule, inputs):
    a, b, c = (x.double() for x in inputs)
    if rule == "DIV-MUL-RCP":
        if bool((b == 0).any()):
            raise NumericEvent("INCONCLUSIVE", "sampled b == 0; distribution was not conditioned")
        return a / b
    if rule == "MUL-RCP-CANCEL":
        if bool((a == 0).any()):
            raise NumericEvent("INCONCLUSIVE", "sampled a == 0; inverse cancellation requires nonzero a")
        return torch.ones_like(a)
    if rule == "LOG-MUL":
        if bool(((a <= 0) | (b <= 0)).any()):
            raise NumericEvent("INCONCLUSIVE", "log requires a > 0 and b > 0; no abs, truncation or resampling")
        return torch.log(a * b)
    if rule == "EXP-SUB":
        return torch.exp(a - b)
    if rule == "EXP-ZERO":
        return torch.ones_like(a)
    if rule == "EXP-NEG-INF-SUB":
        return torch.zeros_like(a)
    if rule in {"MAX-COMMUTE", "MAX-ASSOC"}:
        ab = torch.maximum(a, b)
        return torch.maximum(ab, c) if rule == "MAX-ASSOC" else ab
    if rule in {"ADD-ZERO", "MUL-ONE", "DIV-ONE", "LOG-EXP", "MAX-IDEM", "MAX-NEG-INF"}:
        return a
    raise ValueError("unknown supplemental oracle")


def observe(torch, reference, candidate, exact, fmt, errors=None):
    result = original.observe(torch, reference, candidate, exact, fmt["output"])
    if fmt["compute"] == "fp64":
        if errors is None or len(errors) != 2:
            raise ValueError("fp64-work instance requires the bound residual oracle")
        for key, error in zip(("reference_error", "candidate_error"), errors):
            result[key] = error.max().item()
    return result


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
