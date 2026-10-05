"""Independent-seed confirmation of both fixed FP32 log-product alternatives.

The alternate seed and both branch formulas are fixed before inspecting results.
"""
PROFILE = {
    "shape": [4096, 4096],
    "distribution": {"family": "normal", "mean": 1.0, "std": 1.0},
    "seed": 20261005,
    "replicates": 4096,
    "replicates_max": 50000,
    "batch": 512,
    "formats": [
        {"name": "fp32", "input": "fp32", "compute": "fp32", "accumulator": "fp32", "output": "fp32"},
    ],
    "rules": ["LOG-MUL", "LOG-MUL-LOG1P", "LOG-MUL-GUARDED"],
    "launch": {"block": 1024, "num_warps": 4},
    "gates": {
        "bias": {"tau": 0.05, "se_multiplier": 5.0},
        "vars": {"quantile": 0.9, "horizon": 625000, "alpha": 1.35e-3,
                 "bootstrap": 1000, "min_exceedances": 40, "warn": 10.0, "fail": 100.0},
        "warning_policy": "pass_only",
    },
}
