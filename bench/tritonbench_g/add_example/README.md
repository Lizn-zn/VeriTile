# add_example

Source: [add_example.py](add_example.py), TritonBench-G v1.

- [Kernels.lean](Kernels.lean): the complete original masked addition and a
  candidate changing only `x + y` to `y + x`. Both retain symbolic sizes,
  both input loads and the masked output store.
- [Correct.lean](Correct.lean): original and candidate implement the independent
  formula `xs i + ys i`. The IO contract includes termination, active-address
  bounds and preservation of every other memory cell.
- [FPEquiv.lean](FPEquiv.lean): complete program equivalence under fp32
  `add_commute`. Both actual loaded operands must be finite at the addition
  site, including masked lanes whose arithmetic is evaluated. The proof keeps
  all registers and memory and agrees on execution failures.
- [Proofs/Real.lean](Proofs/Real.lean): real execution, arithmetic, bounds and
  memory-frame lemmas. The FP composition stays in its public entry and uses
  `equiv_decompose` followed by `fp_prove`.

The real and FP layers share Kernels.lean and do not import each other.
Element count and block size remain symbolic; the experiment's dimensions
select the admitted atom and do not constrain this derivation. These are
per-program contracts; host grid composition remains outside their scope.
The candidate illustrates a rewrite, with no performance or bitwise IEEE claim.

Coverage and source fingerprints: [coverage_review.json](../coverage_review.json).
