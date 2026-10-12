# chunk_cumsum_vector

**Padding repair accepted:** both complete real sources now compute valid
column prefixes and preserve all other cells for arbitrary unspecified values
on both axes. FP equivalence still uses only `add_commute`. Compilation,
regressions and independent replay pass; see the receipt below and
[the shared padding review](../block_padding_review.md).

Source: `chunk_cumsum_vector.py`, `chunk_global_cumsum_vector_kernel`.

- `Kernels.lean`: complete typed original and sum-left candidate. Only the
  post-store `b_z += tl.sum(b_s, 0)` changes operand order. Both sources retain
  the lower-triangular dot, `allow_tf32=False`, fp32 input promotion, output
  storage conversion, both boundary axes, the loop and the source branch.
- `Correct.lean`: both full sources implement an independent per-column prefix
  sum over initial inputs after precision erasure. T/S, BT/BS and all three
  strides remain symbolic; every chunk and both partial tails are covered.
  The IO footprint is the logical T-by-BS output window, although execution
  uses BT-by-BS blocks. The contract requires positive BT, distinct input/output,
  disjoint allocations, sufficient active-address bounds, nonoverlapping active
  output addresses across all chunks, and the shared flat adapter's zero-oracle
  condition. It guarantees successful flat execution, every active output and
  preservation of all other tagged cells. `RealPadding.source_outputs` also
  proves complete region execution, independent prefixes and the full cell
  frame for arbitrary unspecified real values. BS need not be positive.
- `FPEquiv.lean`: complete typed equivalence under `add_commute`. The domain
  follows the reached store and branch, then checks the actual fp32 carry and
  block sum. Dot and reduction remain opaque, with no triangular-dot identity
  or reduction association assumed in the FP proof.
- `Proofs/`: real projection, independent prefix partition, initialization,
  complete-loop invariant, active-address safety, flat IO, real candidate
  transfer, arbitrary-padding source correctness and FP composition. `LegacyReal.lean` retains the earlier single
  block/store-slice proofs and the reused triangular-dot lemma; those old
  headlines do not describe the new public contracts.

`Proofs/TypedExecution.lean` checks a consistent abstract backend with bf16
input, fp16 output, nonzero feature/head IDs, T=5/S=3/BT=2/BS=2, three time
chunks and both tails. It checks both complete runs, active outputs and nearby
untouched cells. This witnesses inhabited FP premises, not GPU numerical
behavior. `FPVectorCumsumSource.lean` checks casts, TF32 and both boundary axes.
`VectorCumsumLayout.lean` proves ordinary contiguous matrices satisfy the
address condition and rejects a zero time stride that aliases different rows.
`VectorCumsumPadding.lean` checks both complete real sources with arbitrary
padding, including seven, at T=5/S=3/BT=2/BS=2 and nonzero feature/head IDs.
It verifies all five active prefixes and untouched neighboring tagged cells.
The final carry may include tail values; every valid prefix only includes
valid earlier rows in its own column, so those values cannot affect it.

The source audit expands the literal `original` macro branch and resolves
Python dtype attributes from explicit region/load bindings. Those lexical
checks complement source review and Lean AST checks; they are not a verified
Python compiler. Host launch composition, fixed-width address overflow,
concrete IEEE equality and GPU performance are separate claims.

Validation after the padding repair: the real/FP/typed-execution build passes
3,458 targets. Source, layout and complete padding regressions pass. Independent
comparator/trust replay accepts all seventeen case files and three regressions:
20 files, 96 named proof targets and 927 kernel theorem targets;
all 1,159 snapshot inputs match the checked worktree (`Logs/comparator-check-gcqf5awz`). Coverage review
and the proof-gap manifest pass. Historical receipts are retained in
`../verification_round.md`; no whole-corpus build success is claimed.
