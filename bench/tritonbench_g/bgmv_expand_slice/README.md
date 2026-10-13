# bgmv_expand_slice

**Reopened: FP precision correction.** The earlier source expansion widened
half/bfloat16 sums to fp32. The corrected source retains the product's sum dtype
and specializes the output dtype before arithmetic promotion. Both complete
real correctness specifications rebuild successfully; the old general FP theorem
and fp16-output execution witness still need updating. The source precision
regression is refreshed and passes. In particular,
fp16 addition has no admitted `add_commute` row in the current tables. This case
is not currently counted as complete, and its older receipt does not validate
the new FP lowering.

`Proofs/SupportedFP32.lean` now proves the unchanged output-first candidate for fp32
output storage, across every input/weight/cast choice and both masking and
accumulation flags. `Proofs/FP32.lean` checks finite actual promoted operands
under the taken sentinel branch, reached output iteration and accumulation arm.
The typed reduction and explicit output conversion remain unchanged. Its only
numerical atom is the existing admitted fp32 `add_commute`; this scoped theorem
does not discharge the arbitrary-output theorem in `FPEquiv.lean`.
`Proofs/FP32Execution.lean` supplies a separate nonempty execution and guard
witness for this supported theorem: fp32 input cast to bf16 weights, fp32
output, a masked K tail, two output-loop iterations and signed-sentinel skip.
It uses the consistent abstract backend, not a GPU experiment.

The admission gap is specifically fp16: its addition has no admitted row.
There are existing bf16 and bf16-to-fp32 commutation report rows, but the old
`ScalarArithmetic.Rules` interface exposes only fp32 atoms, so applying those
rows would require a separate correctly typed and validated rule interface.
The old theorem and its signature are retained as pending, without new axioms,
placeholder proofs or a substituted candidate.

`Proofs/GeneralCountermodel.lean` checks that the old fp32-only assumptions
cannot prove the general target. It copies the pending domain locator exactly
and proves source correspondence with `TypedSource.original` and `outputFirst`.
For fp16 input/weights/output, K=N=BK=BN=SPLIT_N=1, unit strides, active slot 0
and accumulation enabled, both complete kernels execute. A model changing only
fp16 addition to left projection satisfies every selected fp32 scalar rule;
the original stores 1 at Y[0], while the candidate stores 2. The copied locator
returns an empty domain for this source. `not_output_first_equiv` refutes the
old equivalence for every `ScalarArithmetic.Rules` parameter. Its axiom audit
contains only `propext`, `Classical.choice` and `Quot.sound`. This is a logical
countermodel to the current assumptions, not an IEEE or GPU numerical failure.
The original public declarations remain unchanged and pending.

`Proofs/AdditionProfiles.lean` adds separate typed scalar admission plumbing
for the existing bf16, bf16_fp32 and fp32 ADD-COMMUTE report rows. Input guards
read each profile's typed registers and observations read its typed output.
Validation remains an explicit premise for the exact bound fragments; report
metadata and admission are checked without adding axioms. Inspection of
`experiments/floating_point/kernels.py` shows that both bf16 profiles widen
inputs to fp32 before addition. The bf16 profile then rounds through
bf16→fp32 before the final bf16 output conversion; bf16_fp32 converts the fp32
result directly to output bf16. The new fragments preserve these conversions.
Consequently these rows alone do not establish commutativity of the abstract
native bf16 primitive used by BGMV: an explicit lowering correspondence is
still required. No fp16 or fp64 report is supplied, and this plumbing is not
connected to or counted as a proof of the pending general theorem.

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
choices have full-source real correspondence; the refreshed source regression
checks the corrected typed sum, promotion and conversion placement.

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
