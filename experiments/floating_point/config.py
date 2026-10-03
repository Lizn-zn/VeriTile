"""Edit this file, then run scripts/check_numerics.py run --output <new-directory>.

std is sigma, NOT sigma squared. Each replicate is a fresh entire input tuple.
These are example parameters, not measured or calibrated acceptance results.
"""

PROFILE = {
    "shape": [4096, 4096],
    "distribution": {"family": "normal", "mean": 1.0, "std": 1.0},
    "seed": 20261003,
    "replicates": 4096,  # minimum before adaptive stopping
    "replicates_max": 50000,
    "batch": 512,
    "formats": [
        {"name": "bf16", "input": "bf16", "compute": "bf16", "accumulator": "fp32", "output": "bf16"},
        {"name": "bf16_fp32", "input": "bf16", "compute": "fp32", "accumulator": "fp32", "output": "bf16"},
        {"name": "fp32", "input": "fp32", "compute": "fp32", "accumulator": "fp32", "output": "fp32"},
    ],
    "rules": "all",  # Only the 14 atomic relations in rules.json.
    "launch": {"block": 1024, "num_warps": 4},
    "gates": {
        "bias": {"tau": 0.05, "se_multiplier": 5.0},
        "vars": {"quantile": 0.9, "horizon": 625000, "alpha": 1.35e-3,
                 "bootstrap": 1000, "min_exceedances": 40, "warn": 10.0, "fail": 100.0},
        "warning_policy": "pass_only",
    },
}
