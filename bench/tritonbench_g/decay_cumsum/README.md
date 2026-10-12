# Decay cumulative scans

- `Kernels.lean`: the full forward scan, Q/K preparation and reverse gradient
  scan from `decay_cumsum.py`, with their index-commuted candidates.
- `Correct.lean`: six real specifications, explicit finite IO windows and
  independent formulas. Execution and loop proofs are in `Proofs/`.
- `FPEquiv.lean`: three complete FP program equivalences. The only change is
  `i_k * BK` to `BK * i_k` inside the lane mask; all three print `none`.

The forward formula is `1.44269504 * sum(g[0..t])`. Preparation computes
`q[t] * exp2(g[t]) * scale` and `k[t] * exp2(g[last] - g[t])`.
Backward computes the corrected dq/dk streams and the reverse cumulative sum
of `dq[t] * q[t] - dk[t] * k[t]`, all from initial input arrays.
These are per-program real formulas after precision erasure.

Storage supports fp16, bf16 and fp32. Preparation and backward permit
independent data and gate storage. Source fp32 accumulators/conversions,
exp2 operations, loop order and final casts are retained. Preparation loads
all lanes of its final gate row without a mask; the IO read window includes
these extra cells. Backward retains its final pointer decrements and updates
dq/dk arrays in place.

The real contracts require `BK <= DK`, and preparation/backward also require
`BT > 0`. Their explicit region separation conditions protect input values
from earlier stores. Each contract also requires a finite disjoint allocation,
bounds for actual accesses and a zero unspecified-value oracle, and proves
successful full flat execution, active output formulas and preservation of
all other tagged cells. These conditions are visible at the real entry and
in its `Implements` definitions. Shapes and strides remain symbolic.

Finite real witnesses cover every storage pair with two iterations, nonzero
program IDs and a partial lane mask. Separate abstract FP executions from
empty registers witness that the full typed programs and candidates can run;
they are not GPU execution measurements. The FP structural equivalences
preserve complete states and failures and need no numerical atoms or extra
numerical-domain restrictions. No host launch, concurrent GPU execution,
IEEE rounding or fixed-width index-overflow guarantee is claimed.

Validation passed: complete repository closure, all-three-JIT source expression
checks, precision regressions, source-only corpus gates, 31 audit/layout tests
and independent comparator/trust replay of all 27 files. Final target inventories
and source fingerprints match `Logs/comparator-check-d9lunxqu`.
