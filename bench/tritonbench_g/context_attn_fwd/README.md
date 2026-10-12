# context_attn_fwd

- Source: `context_attn_fwd.py::_fwd_kernel_int8kv`. The supplied test uses floating Q/K/V storage; the function name does not add an int8 conversion.
- `Kernels.lean`: full original and scale-first candidate, for fp16/bf16/fp32 storage. The only change swaps `qk * sm_scale` inside the `where` arm.
- `Correct.lean`: both versions equal an independent base-2 weighted sum in real semantics. The finite `-1e8` causal sentinel and padded denominator weights are included. The contract proves successful finite flat execution, active outputs and preservation of every untouched tagged cell.
- `FPEquiv.lean`: complete source equivalence under admitted `mul_commute`, with finite actual fp32 qk/scale operands at every reached iteration. Every lane in the eager arm is checked. It preserves binary dot, fused accumulator-dot, exp2, masks, casts, stores and failure.

The real contract keeps dimensions, strides, grouped KV heads, program IDs and
signed metadata symbolic. It requires positive BN, valid actual addresses and
injective active output addresses. Inputs may alias output: all loads precede
the sole store. Experimental shapes do not restrict the symbolic derivation.

`Proofs/` contains execution, loop invariants, numerical algebra and memory
proofs. `RealWitness.lean` supplies a finite three-block execution with grouped
heads, a cached prefix, a partial final key block and no preinitialized
registers. `TypedExecution.lean` supplies reached FP operands and an actual
store under a consistent abstract model. These are Lean-model witnesses,
without a claim of GPU validation, IEEE equality or concurrent-launch safety.

Validation passed: affected libraries and 38-case build, five Lean regressions, source
and scope gates, and 71 independent comparator/trust replays. Final inventories
and source/configuration fingerprints match `Logs/comparator-check-xynmtro4`.
