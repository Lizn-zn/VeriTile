# bgmv_expand_slice

**Reopened: FP precision correction.** The earlier source expansion widened
half/bfloat16 sums to fp32. The corrected source retains the product's sum dtype
and specializes the output dtype before arithmetic promotion. Both complete
real correctness specifications rebuild successfully; the old FP theorem,
its source regression and execution witness still need updating. In particular,
fp16 addition has no admitted `add_commute` row in the current tables. This case
is not currently counted as complete, and its older receipt does not validate
the new FP lowering.

The three public entries share the complete source of
`bgmv_expand_slice.py`'s `_bgmv_expand_slice_kernel`:

- `Kernels.lean`: original and candidate that swaps the reduction result and
  loaded old output in the final addition. All input dtypes (fp16/bf16/fp32),
  both weight dtypes, explicit output dtype and both CAST_TYPE choices are
  represented, along with both EVEN_K and ADD_INPUTS branches.
- `Correct.lean`: both precision-erased sources implement an independent full
  rank-K dot product, plus **initial output** when accumulating. The contract
  includes successful flat-memory execution, actual access bounds and full-cell
  preservation. Initial vector, weights and old output are separate input
  values; the output allocation is listed once.
- `FPEquiv.lean`: the previous fp32 addition proof, awaiting replacement by
  the actual dtype-specific rule and domain. The intended checks still follow
  the non-sentinel branch, reached output-loop iteration and accumulation arm.

The real contract states K ≤ BK, positive BN, output stride and split count,
separate weight/output regions, disjoint allocations, initial input pins and
zero undef. Signed metadata is either -1 (skip) or nonnegative. Shapes and
strides remain symbolic. The input vector is loaded even at zero output length
on an active path, so its bounds are independent of output-loop count. The
source has no global N mask: all computed split rows need storage. Its output
base is `batch*outputBatchStride + split*length + slice*outputStride`; the split
term has no output-stride multiplier.

The source macro stages CAST_TYPE before register typing, so a dtype-changing
cast is retained. Output dtype is also specialized. The source's implicit
output-pointer conversion is explicit at each branch's accumulator assignment,
with no arithmetic between that conversion and the common store. This gives
both branches the actual stored dtype without widening the sum itself.
The Python early return is a non-sentinel active branch. These translation
choices have full-source real correspondence; the FP source regression must
be refreshed for the corrected typed sum and conversion placement.

`Proofs/RealIOExecution.lean` instantiates the complete contract with four
concrete disjoint allocations, nonzero program IDs and slot, N=5/SPLIT_N=2,
K=3/BK=4, BN=2, nonunit strides, negative input, nonzero input padding and a
masked output tail. All dtype/cast/load choices and both versions produce
141/172/203 when accumulating and 30/58/86 otherwise. Only flat cells 91/94/97
may change. Masked weight padding is outside the weight allocation; the actual
active accesses remain in bounds. The signed sentinel preserves all memory.

`Proofs/TypedExecution.lean` is the earlier abstract-model witness and still
needs the typed-sum backend and new guard path. It is not a GPU numerical
experiment, and is not counted as current validation until rebuilt.
`Proofs/RealFoundation.lean` contains the generic tile/mask/pointer lemmas
used by the current proof. `RealSetup` executes the source's full signed
base-offset sum and proves it equals the displayed read window on the
nonnegative active slots required by the public contract.
`Proofs/LegacyClippedOffset.lean` retains the historical explicit AST and its
proofs separately. That AST clipped the signed slot product before adding the
split offset, which disagrees with the source for some negative non-sentinel
slots. Its checked counterexample uses `3*(-2)+10`: old address 10, source
address 4. Current correctness proofs do not import the historical model.

Host launch composition, fixed-width address overflow and concrete IEEE
identities are outside these per-program contracts. Validation status is
recorded in `../verification_round.md`.
