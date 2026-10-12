# adam_update_triton

Source: [adam_update_triton.py](adam_update_triton.py), TritonBench-G v1.
The source name says Adam, but the computation is Lion: one momentum buffer,
a sign-based parameter update and decoupled weight decay.

- [Kernels.lean](Kernels.lean): complete original `update_fn_kernel` and
  `update_commuted_kernel`. The candidate only changes the final momentum
  addition from `diff * beta2 + grad` to `grad + diff * beta2`. Both retain the
  original masks, pointer calculations, comparisons and two in-place stores.
- [Correct.lean](Correct.lean): both sources implement the independent formulas
  `p' = (1 - lr * wd) * p - lr * sign(beta1 * m + (1 - beta1) * g)` and
  `m' = beta2 * m + (1 - beta2) * g`, using the initially loaded values.
  The parameter and momentum output regions must differ. The contracts include
  termination, active-address bounds and the memory frame outside both windows.
- [FPEquiv.lean](FPEquiv.lean): complete fp32 program equivalence using only
  `add_commute`. The computed `diff * beta2` and loaded `grad` must be finite
  at the final momentum addition, including evaluated masked lanes. Their sign
  and zero status are unrestricted. Comparisons retain their original meaning;
  when a backend comparison is unavailable, both executions fail identically.
- [Proofs/Real.lean](Proofs/Real.lean): execution, value, frame and pointer-safety
  lemmas. The FP proof uses `equiv_decompose` and `fp_prove` in its public entry.

`Source.prefixBefore ... "exp_avg" 1` locates the second assignment to momentum,
after the load and all update computations. It does not depend on statement
indices. The FP proof preserves the whole state, both stores and execution
failure, even when the output regions alias; the independent real output
formulas require those regions to differ because the second store could
otherwise overwrite the first output.

The proof layers share Kernels.lean and do not import each other. Dimensions,
scalar parameters and region names stay symbolic. These are per-program results;
host launch composition and autotuning are outside the contract. The FP result
uses the experimentally admitted atom and makes no IEEE or performance claim.

Current scopes and source fingerprints: [coverage_review.json](../coverage_review.json).
