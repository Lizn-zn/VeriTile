"""Bounded count conversion probes; the interval is half-open, in integers.

Only row lengths N <= 2**24 are covered by the successor experiment.
The floating arithmetic, output, and gate budgets remain fp32.
"""

PROFILE = {
    "shape": [4096, 4096],
    "distribution": {"family": "uniform_integer", "low": 0, "high": 16777216},
    "seed": 20261003,
    "replicates": 4096,
    "replicates_max": 50000,
    "batch": 512,
    "formats": [
        {"name": "int32_fp32", "input": "int32", "compute": "fp32", "accumulator": "fp32", "output": "fp32"},
    ],
    "rules": ["COUNT-ZERO", "COUNT-SUCCESSOR"],
    "launch": {"block": 1024, "num_warps": 4},
    "gates": {
        "bias": {"tau": 0.05, "se_multiplier": 5.0},
        "vars": {"quantile": 0.9, "horizon": 625000, "alpha": 1.35e-3,
                 "bootstrap": 1000, "min_exceedances": 40, "warn": 10.0, "fail": 100.0},
        "warning_policy": "pass_only",
    },
}
