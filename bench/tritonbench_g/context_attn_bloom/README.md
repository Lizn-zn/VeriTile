# context_attn_bloom

Source: `context_attn_bloom.py::_fwd_kernel` (TritonBench-G v1).
The complete original and scale-first candidate are in [Kernels.lean](Kernels.lean).
The candidate commutes only `qk * sm_scale`.

- [Correct.lean](Correct.lean): two real specifications, visible IO and the
  independent exponential prefix-sum formula.
- [FPEquiv.lean](FPEquiv.lean): complete typed program equivalence; the only
  printed atom is `mul_commute`.
- [Proofs/](Proofs/): source alignment, staged execution, loop invariant,
  arithmetic, actual-access bounds, finite flat IO and execution witnesses.

Both versions retain fp16/bf16/fp32 storage, signed int32 metadata and page
indices, grouped KV heads, the prompt cache, gathered pages, partial time and
channel masks, and both probability/output conversions. The dynamic
probability cast and implicit output conversion are explicit at the selected
storage dtype. `tl.exp` is unchanged on both sides; no exp rewrite is assumed.

The real result includes the source's finite `-1e8` sentinel and separate
`acc_scale` guard. Future lanes therefore have nonzero mathematical exponential
weights. The specification is a guarded normalized-block recurrence, not ideal
causal softmax. [Proofs/FiniteMaskFormula.lean](Proofs/FiniteMaskFormula.lean)
checks a concrete difference between those mathematical formulas; it is not a
GPU observation or an independently executed source counterexample.

The real contract requires `BN > 0`, nonnegative prompt length, injective active
output addresses and nonnegative in-bounds actual addresses in a finite disjoint
allocation. Shapes, strides and launch IDs remain symbolic. All loads precede
the only store, so inputs may alias output. Successful execution, the output
formula and preservation of every other tagged cell are proved together.
No precomputed output or zero unspecified-value premise is required.

The FP domain checks finite actual qk/scale operands at every reached scaling
site. Its proof preserves all other operations, memory effects and failures.
[Proofs/TypedExecution.lean](Proofs/TypedExecution.lean) supplies full original
and candidate executions under a named consistent abstract model, with active
and masked output observations. This witnesses nonempty assumptions; it does
not assert IEEE arithmetic or measured GPU behavior.
[Proofs/RealWitness.lean](Proofs/RealWitness.lean) gives a finite-allocation real
execution with a cached prefix, grouped head, nonzero block and three visited
key blocks including a partial tail. These constants do not constrain the
symbolic specifications. Source precision and signed-inference regressions
are in `bench/tests/FPContextAttnBloomSource.lean`.

Validation passed: repository entries, proof closure and witnesses compile;
typed/signed source regression, source/coverage gates and all 68 independent
comparator/trust checks pass. Final target inventories and source fingerprints
match the checked snapshot at `Logs/comparator-check-u0a7p72i`. This is per-program verification,
without a host-launch, fixed-width index-overflow or performance claim.
