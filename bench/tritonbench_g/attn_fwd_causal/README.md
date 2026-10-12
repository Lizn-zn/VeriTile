# attn_fwd_causal

Source: `attn_fwd_causal.py` (TritonBench-G).

- [Kernels.lean](Kernels.lean) contains the complete original and reciprocal
  candidate. The only candidate change is final division to multiplication
  by the reciprocal.
- [Correct.lean](Correct.lean) displays the actual address layout, two IO
  contracts and the independent normalized base-2 weighted-sum formula.
  Both complete erased sources successfully execute in flat memory, produce
  this formula on active output lanes and preserve all other cells.
- [FPEquiv.lean](FPEquiv.lean) compares the complete typed sources using only
  `div_mul_rcp`. Its guards check the actual fp32 accumulator and nonzero row
  denominator after both helper loops. No input-sign restriction or output
  equation is assumed.

Outer STAGE=3 visits the prefix then the diagonal; STAGE=2 visits the diagonal
**twice**. Each helper starts from the caller's original pointer bases. Future
diagonal scores receive a finite `-1e6` additive penalty, so their real weights
remain positive. Masked K/V values are zero, but their weights still count in
the denominator. The mathematical formula preserves these source behaviors.

Dimensions and independent strides remain symbolic. The real IO requires
positive key-block size, disjoint bounded allocations, actual-read input
pins, zero undefined masked loads and distinct active output addresses. It
retains the source's use of Q strides for V/Out and unmasked Q/K scale loads,
including scales for entirely padded tiles. There is no scratch buffer.
`bench/tests/RealAttnFwdCausalIO.lean` checks partial, unaligned ranges and the
scale extent required by the last fully masked tile.

The FP sources retain fp16 Q/K/V/Out, fp32 scales/accumulators, both dot
precisions, the probability cast, exp2, masks and stores. The numerical domain
covers every lane evaluated at normalization, including masked output lanes.
[Proofs/TypedExecution.lean](Proofs/TypedExecution.lean) supplies a consistent
abstract-model execution/domain witness for both stages. Experimental shapes
select the admitted atom; they are not proof dimensions or new GPU results.

[Proofs/RealFlatCounterexample.lean](Proofs/RealFlatCounterexample.lean) refutes
exact causal masking for the actual real source: a concrete disjoint flat
allocation with Q/K=0, scales=1 and V=[0,1] successfully produces a positive
first output in both versions. The small symbolic fixture is a Lean real
counterexample, not a GPU launch or an fp16-underflow claim.

[Proofs/LegacyExactMask.lean](Proofs/LegacyExactMask.lean) preserves the old
changed model (one full-context loop, replaced scores, zeroed probabilities
and ignored STAGE). Its proofs are historical and no longer appear in the
public specification inventory.

Both helpers are explicitly inlined; Python head constants 128/96 become
BLOCK_DMODEL/HEAD_ACTIVE. Other outer stages, whose Python helper bounds are
undefined, are excluded. Host launch composition and fixed-width address
overflow are outside the per-program statements.

Final validation: `TritonBenchGVerified` builds (3,723 jobs); both focused
source/IO regressions, all 29 metadata/layout/shard tests and all source gates
pass. Independent comparator/trust replay accepts all 48 case/regression
files, including 307 named proofs and 1231 kernel theorem targets;
all 824 snapshot input hashes match. Evidence: `Logs/comparator-check-sve03cl2`.
