"""Edit this file, then run scripts/fp_experiment.py run --output <new-directory>.

std is sigma, NOT sigma squared. Each replicate is a fresh entire input tuple.
These are example parameters, not measured or calibrated acceptance results.
"""

PROFILE = {
    "shape": [4096, 4096],
    "distribution": {"family": "normal", "mean": 1.0, "std": 1.0},
    "seed": 20261001,
    "replicates": 4096,
    "formats": [
        {"name": "bf16", "input": "bf16", "compute": "bf16", "accumulator": "fp32", "output": "bf16"},
        {"name": "bf16_fp32", "input": "bf16", "compute": "fp32", "accumulator": "fp32", "output": "bf16"},
        {"name": "fp32", "input": "fp32", "compute": "fp32", "accumulator": "fp32", "output": "fp32"},
    ],
    "rules": "all",
    "launch": {"block": 1024, "chunk": 256, "num_warps": 4, "dot_tile": 16},
    "layernorm_epsilon": 1e-5,
    "gates": {
        "bias": {"z": 5.0, "snr": 0.01, "ulp_floor": 1.0},
        "vars": {"quantile": 0.9, "horizon": 625000, "confidence_z": 3.0,
                 "bootstrap": 256, "min_exceedances": 64, "warn": 2.0, "fail": 10.0},
        "warning_policy": "pass_only",
    },
}
