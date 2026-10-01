#!/usr/bin/env python3
"""Compile every supported atomic relation without executing kernels or requiring CUDA.

Needs Triton (tested with 3.5.1). Successful compilation is NOT a numerical gate
result. NVIDIA architecture 80 is the default offline compilation target.
"""
import argparse
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", type=int, default=80)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--columns", type=int, default=33)
    args = parser.parse_args()
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    path = Path(__file__).resolve().parents[1] / "experiments/floating_point/kernels.py"
    spec = importlib.util.spec_from_file_location("numerical_kernel_templates", path)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    count = 0

    def compile_kernel(fn, pointers, constants):
        nonlocal count
        source = ASTSource(fn, {**pointers, **{key: "constexpr" for key in constants}}, constants)
        triton.compile(source, target=GPUTarget("cuda", args.arch, 32),
                       options={"num_warps": 4, "enable_fp_fusion": False})
        count += 1

    for dtype, low in (("bf16", True), ("bf16", False), ("fp32", False)):
        ptrs = {name: "*" + dtype for name in ("A", "B", "C", "O")}
        for rule in kernels.ELEMENTWISE:
            if rule == "BF16-WIDEN-RETURN" and dtype != "bf16":
                continue
            for side in (0, 1):
                compile_kernel(kernels.elementwise, ptrs,
                               dict(N=args.rows * args.columns, RULE=rule, SIDE=side, LOW=low,
                                    INPUT_LOW=dtype == "bf16", BLOCK=1024))
        print(f"Compiled {dtype}, low={low}: {count} specializations", flush=True)
    print(f"PASS: {count} offline compilations for sm_{args.arch}; no kernels executed.")


if __name__ == "__main__":
    main()
