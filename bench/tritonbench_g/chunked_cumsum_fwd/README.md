# chunked_cumsum_fwd

Source: `chunked_cumsum_fwd.py`, `_chunk_cumsum_fwd_kernel`.

- `Kernels.lean`: complete typed original and `scaleFirst` candidate. The
  candidate changes only `dt * A[:, None]` to `A[:, None] * dt`.
- `Correct.lean`: both complete kernels equal an initial-array formula for
  prepared dt and its scaled inclusive row prefix sum. Bias, cutoff softplus,
  lower/upper clamping, remasking, both stores and scan are included.
- `FPEquiv.lean`: full typed program equivalence using only `mul_commute` on
  actual reached finite fp32 operands. All other numeric operations stay opaque.
- `Proofs/`: source projection, complete execution, active-address bounds,
  finite flat-memory transport, output/frame proofs and inhabited FP models.

Input dt/A storage is independent; bias has fp32/fp16/bf16 variants and both
outputs are fp32. Upper bounds may be finite or the default `+∞`. The infinite
minimum remains a distinct FP operation; no FP identity is assumed for it.
Dimensions and strides are symbolic. The real contract requires the host-valid
chunk start, active-write injectivity, disjoint finite allocations and actual
access bounds. A must differ from the first output, and the two outputs must
differ. No `undef=0` premise is needed because every load has explicit other=0.

The scope is one launched JIT program. Host allocation/grid construction,
backend legality, cross-program races and performance are separate. The typed
execution witnesses use a consistent abstract backend, not measured GPU values.

`bench/tests/ChunkedCumsumReal.lean` checks nonzero IDs, head/time tails, finite
allocations, both limits and both variants. `ChunkedCumsumExecution.lean` audits
public specifications and full typed executions. Integration and independent
comparator/trust acceptance are recorded in `../verification_round.md`.
