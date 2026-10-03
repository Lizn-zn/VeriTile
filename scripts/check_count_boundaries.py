#!/usr/bin/env python3
"""GPU exhaustive count check and out-of-range counterexample; not a gate sample."""
import argparse
from copy import deepcopy
from pathlib import Path

if __package__:
    from . import check_numerics_supplement as runner
else:
    import check_numerics_supplement as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import triton
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise ValueError("this check requires an NVIDIA GPU")
    profile = runner.validate_profile(deepcopy(runner.load_module(
        runner.ROOT / "experiments/floating_point/primitives/count_config.py").PROFILE))
    fmt = profile["formats"][0]
    kernels = runner.load_module(runner.KERNELS)
    a = torch.arange(0, 2**24, device="cuda", dtype=torch.int32)
    (reference, candidate), compiled, _ = runner.launch_pair(
        torch, triton, kernels, "COUNT-SUCCESSOR", [a, a, a], profile, fmt)
    exact = (a.to(torch.float64) + 1)
    mismatches = int(((reference.double() != exact) | (candidate.double() != exact)).sum().item())
    if mismatches:
        raise ValueError(f"bounded successor failed for {mismatches} integers")
    boundary = torch.tensor([2**24 - 1, 2**24, 2**24 + 1, 2**24 + 2, 2**24 + 3],
                            device="cuda", dtype=torch.int32)
    (ref, cand), _, _ = runner.launch_pair(
        torch, triton, kernels, "COUNT-SUCCESSOR", [boundary] * 3, profile, fmt)
    if (ref[2].item(), cand[2].item()) != (16777218., 16777216.):
        raise ValueError("counterexample disappeared; inspect the actual integer conversion lowering")
    data = {
        "hardware": torch.cuda.get_device_name(),
        "audit_source_sha256": runner.sha(Path(__file__).read_bytes()),
        "sources": runner.source_hashes(),
        "exhaustive": {"low": 0, "high_exclusive": 2**24, "tested": a.numel(), "mismatches": mismatches},
        "lowerings": {side: [runner.sha(ptx.encode()) for ptx in programs] for side, programs in compiled.items()},
        "boundary": [{"i": i, "reference": r, "candidate": c, "equal": r == c}
                     for i, r, c in zip(boundary.tolist(), ref.tolist(), cand.tolist())],
        "scope": "GPU exhaustive bounded check and negative control; these are not extra IID gate replicates",
    }
    runner.write_json(args.output, data)
    print(f"All {a.numel()} bounded integers match exactly; the out-of-range counterexample is preserved.")


if __name__ == "__main__":
    main()
