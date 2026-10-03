# Primitive experiments

The [current table](report/summary.md) contains EXP-SUB-INTRINSIC, COUNT-ZERO
and COUNT-SUCCESSOR, with z, B, U and acceptance. DLC job `dlc1qojvuygw2t0b`
ran on H200 (runtime name L20X), using the task name
`traces_kernel_equivalence_testing`. Independent CPU replay exactly matches
the GPU-environment tables.

EXP-SUB-INTRINSIC compares `tl.exp(a-b)` with `tl.exp(a)/tl.exp(b)` in fp32.
Its operands are independent Normal(1,1). It fails the bias budget:
B=0.1608954387 ULP > 0.05, while U=3.958463781 passes. The existing EXP-SUB
uses libdevice.exp and is a separate implementation.

COUNT-ZERO compares an int32 literal zero converted to fp32 with fp32 literal
zero. Constant folding is allowed and recorded in PTX. Repeated executions
do not add stochastic coverage for a constant expression.

COUNT-SUCCESSOR compares `(i+1).to(fp32)` with `i.to(fp32)+1` for uniform
int32 inputs `0 <= i < 2^24`. The reference adds in int32 before conversion;
the candidate converts first and adds in fp32. A separate exhaustive GPU
check covers all 16,777,216 integers with no mismatch. The negative control
at i=16,777,217 produces 16,777,218 versus 16,777,216. Acceptance therefore
supports the bounded count relation only, and any Welford use requires
N <= 2^24. Both count probes have zero observed error, z=B=U=0; their U
uses empirical-max fallback rather than a fitted tail bound.

All three gate runs use shape 4096x4096, R=4096, tau=0.05 local ULP, five
standard errors, and magnitude thresholds 10/100. The accepted count rows
are now bound by `Float/CountConversion`: the successor fragment returns the
same zero on both sides outside the admitted integer range. Extracting the
conversion law requires `i < upperExclusive`, so Welford and LayerNorm retain
`N <= 2^24`. Welford's FP contract also requires a nonempty row; LayerNorm
covers empty output rows. No whole-recurrence atom is introduced.

The dedicated exporter preserves this range and validates the report metadata;
the generic scalar exporter still rejects count rules:

```bash
python3 scripts/export_count_rules.py --trust-report --check
python3 -m unittest scripts.test_count_admission scripts.test_fp_counts -v
```

## Reproduce

Use a fresh output directory for each run. The supplemental runner shares
the main bias/magnitude implementation and binds input types, sampling range,
source hashes, PTX, observations and stopping points into replayable evidence.

```bash
python scripts/check_numerics_supplement.py run --rules EXP-SUB-INTRINSIC --formats fp32 --output Logs/primitive-exp
python scripts/check_numerics_supplement.py report Logs/primitive-exp --output-dir Logs/primitive-exp-report
python scripts/check_numerics_supplement.py run --profile experiments/floating_point/primitives/count_config.py --output Logs/primitive-counts
python scripts/check_numerics_supplement.py report Logs/primitive-counts --output-dir Logs/primitive-counts-report
python scripts/check_count_boundaries.py --output Logs/primitive-count-boundaries.json
```

Add `--smoke` to either run command for four 32x33 replicates, which cannot
be admitted. Local checks:

```bash
python -m unittest scripts.test_primitive_numerics scripts.test_numerics_supplement scripts.test_numerical_domains
python scripts/check_supplement_kernels.py --arch 90 --rules EXP-SUB-INTRINSIC --formats fp32
python scripts/check_supplement_kernels.py --arch 90 --profile experiments/floating_point/primitives/count_config.py
```

`scripts/publish_primitive_results.py RUN` independently replays the DLC
layout (`exp/atomic`, `counts/atomic`, their GPU report directories,
`count-boundaries.json`, `status.json`, and `exit_code`) and updates this
directory's current report. Raw observations and PTX remain outside Git.
