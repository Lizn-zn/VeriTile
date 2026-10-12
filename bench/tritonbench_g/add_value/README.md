# add_value

Source: [add_value.py](add_value.py), TritonBench-G v1.

- [Kernels.lean](Kernels.lean): the original `puzzle1_kernel` and a candidate
  swapping `x + value` to `value + x`. Both retain the masked load and store.
  The candidate's Lean `Real` annotation selects the scalar data channel and
  introduces no numerical cast. Element count, block size and value are symbolic.
- [Correct.lean](Correct.lean): both sources implement the independent real
  formula `xs i + value`, including termination, active-address bounds and
  preservation of every other memory cell.
- [FPEquiv.lean](FPEquiv.lean): fp32 program equivalence using `add_commute`.
  Both the actual loaded tile and the interpreted scalar must be finite at
  the addition site, including masked lanes whose arithmetic is evaluated.
  Negative and zero values are allowed. The proof preserves every register,
  memory cell and execution failure.
- [Proofs/Real.lean](Proofs/Real.lean): real execution, bounds and frame lemmas.
- [Proofs/LegacyRounding.lean](Proofs/LegacyRounding.lean): preserves the old
  `puzzle1_kernel_io_correctness` theorem as a legacy lemma. Its `.real` store
  has identity boundary rounding, so it is an abstract real-execution result.
  It is not the current FP-equivalence specification. Its explicit legacy IO
  has the same fields as the public real contract; no entry file is imported.

The current real and FP layers share Kernels.lean and do not import each other.
The FP proof uses `equiv_decompose` and `fp_prove`; its scalar-broadcast bridge
uses the existing admitted atom and does not require another GPU experiment.
The contracts are per program; host grid composition, bitwise IEEE identities
and performance are outside their scope.

Current public scopes and source fingerprints are in
[coverage_review.json](../coverage_review.json). The legacy rounding theorem
remains independently replayed with every file under Proofs/ and is registered
separately in [kernel-manifest.tsv](../../../scripts/kernel-manifest.tsv).
