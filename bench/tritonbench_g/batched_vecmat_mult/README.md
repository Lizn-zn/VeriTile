# batched_vecmat_mult

The three public entries share the complete fp32 Triton source from
`batched_vecmat_mult.py`:

- `Kernels.lean`: original and candidate that swaps the two operands of each
  `vecmat` addition. Broadcast, reduction axis, transpose, loop and stores stay
  the same.
- `Correct.lean`: both precision-erased sources satisfy the independent
  `out[m,n] = sum_k A[m,k] * B[m,n,k]` formula, with successful flat-memory
  execution, address bounds and the complete memory frame. Dimensions remain
  symbolic; K=BK*T, BK>0 and BN≤N, with disjoint bounded allocations and the
  displayed unmasked input/output windows.
- `FPEquiv.lean`: complete-program equivalence under admitted `add_commute`.
  Each reached addition requires finite actual accumulated and block-sum
  operands. The fp32 profile, loads, reduction, store and failure behavior are
  preserved; experiment sizes do not restrict the derivation.

`Proofs/` contains the execution, source correspondence and supporting lemmas.
`RealExecution.lean` instantiates both full IO contracts with disjoint memory,
nonzero program IDs, a 2×3 output tile and two K blocks. `TypedExecution.lean`
checks successful complete executions and satisfiable guards in a consistent
abstract model. It is not a GPU numerical experiment.

Host launch composition, fixed-width address overflow and concrete IEEE
identities are outside these per-program contracts. Validation status is
recorded in `../verification_round.md`.
