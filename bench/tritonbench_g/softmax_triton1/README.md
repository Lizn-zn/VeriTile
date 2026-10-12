# softmax_triton1

Source: [softmax_triton1.py](softmax_triton1.py), TritonBench-G v1.

- [Kernels.lean](Kernels.lean): original masked row softmax and the reciprocal
  candidate. Both retain the original strides, masks, `tl.exp`, max/sum order
  and output store. The candidate replaces `numerator / denominator` with
  `numerator * (1 / denominator)`.
- [Correct.lean](Correct.lean): both implementations satisfy the independent
  real formula `exp(xs i) / ∑ j < n_cols, exp(xs j)` within `BLOCK_SIZE`.
  The IO contract includes termination, active-address bounds and preservation
  of all cells outside the masked output row. Any positive block size is
  allowed; the whole row is covered when `n_cols ≤ BLOCK_SIZE`.
- [FPEquiv.lean](FPEquiv.lean): fp32 contextual equivalence using `div_mul_rcp`.
  Every computed numerator lane must be finite, and the scalar row sum finite
  and nonzero, checked at the actual division site. This includes masked lanes
  whose arithmetic is evaluated. No exp identity or positivity premise is used.
- [Proofs/Real.lean](Proofs/Real.lean): execution, real arithmetic, reduction,
  bounds and memory-frame lemmas. The FP proof is short enough to stay in its
  public entry: `equiv_decompose` followed by `fp_prove`.

The FP conclusion preserves the complete final state and agrees on failed
executions. It is conditional on the admitted atom, not a bitwise IEEE or
performance claim. The real and FP layers share only Kernels.lean; neither
imports the other proof layer. The host launch remains outside both contracts.

The per-program coverage category remains `specialization`. Current scopes and
source fingerprints are in [coverage_review.json](../coverage_review.json).
