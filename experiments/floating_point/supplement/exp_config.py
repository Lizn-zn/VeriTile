"""Paired FP32 exp implementations; identical draws and gates within each pair."""

PROFILE = {
    "shape": [
        4096,
        4096
    ],
    "distribution": {
        "family": "normal",
        "mean": 1.0,
        "std": 1.0
    },
    "seed": 20261003,
    "replicates": 4096,
    "replicates_max": 50000,
    "batch": 512,
    "formats": [
        {
            "name": "fp32",
            "input": "fp32",
            "compute": "fp32",
            "accumulator": "fp32",
            "output": "fp32"
        }
    ],
    "rules": [
        "EXP-SUB-INTRINSIC",
        "EXP-SUB",
        "EXP-ZERO",
        "EXP-ZERO-LIBDEVICE",
        "EXP-NEG-INF-SUB",
        "EXP-NEG-INF-SUB-LIBDEVICE"
    ],
    "launch": {
        "block": 1024,
        "num_warps": 4
    },
    "gates": {
        "bias": {
            "tau": 0.05,
            "se_multiplier": 5.0
        },
        "vars": {
            "quantile": 0.9,
            "horizon": 625000,
            "alpha": 0.00135,
            "bootstrap": 1000,
            "min_exceedances": 40,
            "warn": 10.0,
            "fail": 100.0
        },
        "warning_policy": "pass_only"
    }
}
