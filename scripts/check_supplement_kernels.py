#!/usr/bin/env python3
"""Offline compilation of supplemental atoms. This produces no GPU evidence."""
import argparse
from copy import deepcopy

if __package__:
    from . import supplement_numerics as experiment
else:
    import supplement_numerics as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", type=int, default=80)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--columns", type=int, default=33)
    args = parser.parse_args()
    if args.rows <= 0 or args.columns <= 0:
        parser.error("positive dimensions required")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    kernels = experiment.original.load_module(experiment.KERNELS)
    profile = experiment.validate_profile(deepcopy(experiment.original.load_module(experiment.DEFAULT_PROFILE).PROFILE))
    if kernels.SUPPORTED != set(experiment.load_catalog()):
        raise ValueError("catalogue and kernels disagree")
    count = 0
    for fmt in profile["formats"]:
        for rule in profile["rules"]:
            if experiment.unsupported(rule, fmt):
                continue
            for side in (0, 1):
                constants = dict(N=args.rows * args.columns, RULE=rule, SIDE=side,
                                 PRECISION=fmt["compute"], BLOCK=1024)
                pointers = {k: "*" + fmt["input"] for k in ("A", "B", "C")}
                pointers["O"] = "*" + fmt["output"]
                source = ASTSource(kernels.elementwise,
                                   {**pointers, **{k: "constexpr" for k in constants}}, constants)
                triton.compile(source, target=GPUTarget("cuda", args.arch, 32),
                               options={"num_warps": 4, "enable_fp_fusion": False})
                count += 1
            if fmt["compute"] == "fp64":
                constants = dict(N=args.rows * args.columns, BLOCK=1024)
                pointers = {"A": "*fp64", "B": "*fp64", "REF": "*fp32", "CAND": "*fp32",
                            "ER": "*fp64", "EC": "*fp64"}
                source = ASTSource(kernels.quotient_errors,
                                   {**pointers, **{k: "constexpr" for k in constants}}, constants)
                program = triton.compile(source, target=GPUTarget("cuda", args.arch, 32),
                                         options={"num_warps": 4, "enable_fp_fusion": False})
                if "fma.rn.f64" not in program.asm["ptx"]:
                    raise ValueError("residual oracle must retain explicit fp64 FMA")
                count += 1
        print(f"Compiled {fmt['name']}: {count} specializations", flush=True)
    print(f"PASS: {count} offline compilations; no GPU execution or acceptance.")


if __name__ == "__main__":
    main()
