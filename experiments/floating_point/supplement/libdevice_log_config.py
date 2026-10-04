"""Paired fp32 comparisons of tl.log and libdevice.log.

Each libdevice.log rule uses the input seed of its tl.log counterpart.
All sampling, input domains, arithmetic precision and gate budgets match.
"""

PROFILE = {
    "shape": [4096, 4096],
    "distribution": {"family": "normal", "mean": 1.0, "std": 1.0},
    "seed": 20261003,
    "replicates": 4096,
    "replicates_max": 50000,
    "batch": 512,
    "formats": [
        {"name": "fp32", "input": "fp32", "compute": "fp32", "accumulator": "fp32", "output": "fp32"},
    ],
    "rules": ["LOG-MUL", "LOG-MUL-LIBDEVICE", "LOG-EXP-LIBDEVICE", "LOG-EXP-FULL-LIBDEVICE"],
    "launch": {"block": 1024, "num_warps": 4},
    "gates": {
        "bias": {"tau": 0.05, "se_multiplier": 5.0},
        "vars": {"quantile": 0.9, "horizon": 625000, "alpha": 1.35e-3,
                 "bootstrap": 1000, "min_exceedances": 40, "warn": 10.0, "fail": 100.0},
        "warning_policy": "pass_only",
    },
}
