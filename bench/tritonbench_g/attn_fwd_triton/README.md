# attn_fwd_triton

Source: `attn_fwd_triton.py` (TritonBench-G).

- [Kernels.lean](Kernels.lean) contains both complete sources. The candidate
  changes only the final division to multiplication by a reciprocal.
- [Correct.lean](Correct.lean) shows the address layout, two IO contracts and
  independent normalized base-2 weighted-sum formula. Both erased sources
  successfully execute in flat memory, satisfy the formula at active output
  lanes and preserve every other cell.
- [FPEquiv.lean](FPEquiv.lean) proves complete typed-source equivalence under
  `div_mul_rcp`. Its domain checks actual fp32 accumulators and row
  denominators after both helpers: all finite, and denominators nonzero.

The supplied Q/K/V and output have **bf16** storage. Q/K use the fp32-output
score dot; the value dot takes explicit fp16 casts of both probabilities and
V and has fp16 output. Scales and accumulators are fp32. These distinctions
remain in the FP programs, with exp2, masks, stores and failures preserved.
The source regression checks these dtypes and casts in both candidates.
[Proofs/TypedExecution.lean](Proofs/TypedExecution.lean) checks a shared
consistent-model execution/domain witness for both legal stages; it is not a
GPU experiment.

STAGE=3 visits the prefix and then the diagonal; STAGE=2 visits the diagonal
twice. Each helper starts from separate copies of the caller's pointer bases.
Diagonal future-key scores receive a finite additive `-1e6` penalty. Masked
K/V values are zero, but their weights still enter the denominator. The real
formula retains all these source behaviors, including partial and repeated
visits. Python head constants 128/96 become BLOCK_DMODEL/HEAD_ACTIVE.

The real contract requires positive key-block size, disjoint bounded
allocations, values pinned only at actual input reads, zero masked undef and
distinct active output addresses. Q strides determine V and Out addresses;
scale reads remain unmasked, even for fully padded tiles. All dimensions and
source strides remain symbolic. There is no scratch allocation or output
equation as a premise. Host launch composition and fixed-width address
overflow are outside these per-program statements. Outer stages that leave
Python helper bounds undefined are excluded.

[Proofs/RealProjection.lean](Proofs/RealProjection.lean) proves equality of
both complete erased kernels with the two-stage source from attn_fwd_causal.
[Proofs/Real.lean](Proofs/Real.lean) transfers its input windows, independent
formula, successful flat execution and memory frame through that equality.
This reuse applies only to the real layer; FP proofs retain the bf16 source.

[Proofs/RealCounterexample.lean](Proofs/RealCounterexample.lean) checks actual
source execution in a concrete disjoint flat allocation: Q/K=0, scales=1 and
V=[0,1] give a strictly positive first output, refuting exact causal masking
in the real semantics. This small symbolic fixture makes no claim about GPU
launch validity or rounded-output underflow.

[Proofs/LegacyExactMask.lean](Proofs/LegacyExactMask.lean) preserves the old
single full-context loop with replaced scores and zeroed probabilities. Its
statements concern that changed model and have no current public scope rows.

Final validation: `TritonBenchGVerified` builds (3,733 jobs). The bf16 source
regression, refreshed causal source regression, 29 metadata/layout/shard tests
and all source gates pass. Independent comparator/trust replay accepts all
12 case/regression files, including 176 named proofs and
567 kernel theorem targets; all 834 snapshot inputs match.
Evidence: `Logs/comparator-check-_zp921x6`.
