# triton_mul2

Source: [triton_mul2.py](triton_mul2.py), TritonBench-G v1.

- [Kernels.lean](Kernels.lean): both complete originals, `mul2_kernel` and
  `mul2_inplace_kernel`, plus candidates replacing `2 * x` with `x * 2`.
  Every load, mask and store is retained. Element count and block size remain
  symbolic. This is a rewrite example with no performance claim.
- [Correct.lean](Correct.lean): four real specifications, one per source.
  Each states the independent formula `2 * xs i`, including termination,
  active-address bounds and preservation of every other memory cell.
  The ordinary IO has input/output arguments; the in-place IO explicitly wires
  both roles to `ptr` and relates final values to the initially loaded window.
- [FPEquiv.lean](FPEquiv.lean): two fp32 program equivalences, one for each
  original/candidate pair, both using only `mul_commute`. Actual loaded tile
  lanes and the interpreted scalar 2 must be finite at the multiplication site,
  including masked lanes whose arithmetic is evaluated. No positivity or
  nonzero condition is added to the input. The in-place proof keeps the store
  to the same buffer. Both proofs preserve complete state and execution failure.
- [Proofs/Real.lean](Proofs/Real.lean): real execution, safety and memory-frame
  lemmas for both IO variants. The FP proofs stay in the public entry and use
  `equiv_decompose` followed by `fp_prove`.

The real and FP layers share Kernels.lean and do not import each other. The
existing admitted scalar atom is lifted through broadcasting; no new GPU
experiment is needed. The results are per program, without host grid
composition, bitwise IEEE or performance guarantees. Experiment dimensions
select the atom and do not constrain the symbolic proof dimensions.

The per-program coverage category remains `specialization`; precise scopes
and source fingerprints are in [coverage_review.json](../coverage_review.json).
