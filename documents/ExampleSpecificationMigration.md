# Example specification migration

The reference is `TritonBenchVectorAdditionCorrect.lean` and
`TritonBenchVectorAdditionFPEquiv.lean`.

Each completed case has a real correctness file and an independent FP equivalence file.
The correctness statement is `Spec.Real (io ⊨ mathematical_formula)`.
The FP statement is `originalKernel … ≡[R] optimizedKernel …`, with
`#print_fp_assumptions` listing the numerical atoms used by its proof. FP files
define their own kernels and never import their correctness counterpart.

Experimental shape and distribution select assumptions. They are not extra
shape conditions on the subsequent Lean derivation. Kernel dimensions remain
parameters except where a case was already explicitly a fixed-rank slice.

## Coverage ledger

This table tracks the original cases, not just newly compiling files. A real
proof or an unrelated local rewrite does not complete an existing algorithmic
FP equivalence. Pending entries must not be advertised as proved.

| Original case | Correctness file | FP transformation and status |
|---|---|---|
| TritonBench vector addition | `TritonBenchVectorAdditionCorrect` — checked | `TritonBenchVectorAdditionFPEquiv` — checked; add commutation |
| Aligned vector addition | `VectorAddCorrect` — checked | `VectorAddFPEquiv` — checked; add commutation |
| Masked vector addition | `FlatVectorAddCorrect` — checked | `FlatVectorAddFPEquiv` — checked; add commutation |
| Float dtype addition | `FloatDTypeAddCorrect` — checked, including empty tiles | `FloatDTypeAddFPEquiv` — checked; add commutation; output cast retained |
| Row-wise sum | `RowWiseSumCorrect` — checked | Pending reduction-order derivation from scalar atoms |
| Row-wise max | `RowWiseMaxCorrect` — checked | Pending a structural transformation; no max atom is admitted |
| Online softmax | `OnlineSoftmaxCorrect` — checked, original batch-kernel/online-recurrence scope | Batch versus online: pending elementary exp laws and loop/reduction derivation |
| mHC depth | `HyperConnectionsDepthCorrect` — checked, original rank-one/zero-iteration scope | `HyperConnectionsDepthFPEquiv` — checked in the same scope; add commutation |
| mHC width | `HyperConnectionsWidthCorrect` — checked, original rank-one/zero-iteration scope | `HyperConnectionsWidthFPEquiv` — checked in the same scope; two multiplication commutations |
| Adam-named Lion update | `AdamUpdateGridLaunchCorrect` — checked, per-program and grid proofs retained | `AdamUpdateGridLaunchFPEquiv` — checked per program; momentum addition commutation, masked in-place stores retained |
| Stable softmax | `SoftmaxStableCorrect` — checked for both original kernels against the softmax formula | Naive versus stable: missing admitted elementary exp laws |
| Stable logsumexp | `StableLogSumExpCorrect` — checked for both original kernels against logsumexp | Direct versus stable: missing admitted elementary exp/log laws |
| Softmax reciprocal | `SoftmaxReciprocalCorrect` — checked for both original kernels against the softmax formula | Division versus reciprocal multiplication: pending a faithful match to tested `div_rn` |
| Float dtype softmax | `FloatDTypeSoftmaxCorrect` — checked for both original fp32-load/fp64-work kernels against the softmax formula | Original fp64 intermediate arithmetic has no admitted fp64 row |
| Fused SiLU | Pending | Fused versus materialized pipeline: pending structural memory/def-use lemmas |
| Fused SwiGLU | Pending | Fused versus materialized pipeline: pending structural lemmas and explicit rounding-idempotence atom |
| Welford | Pending | Two-pass versus online variance: pending algebra and loop/reduction derivation |
| Fused layernorm | Pending | Two-pass versus online statistics: pending algebra and loop/reduction derivation |

There are currently 14 correctness modules and 7 FP equivalence modules. The
eight legacy equivalence modules remain while their original transformations
are migrated; their presence does not complete the pending FP entries above.

## Admission and proof boundaries

The trusted PR #9 report admits 30 instances of elementary relations. Its
accepted table is unchanged by this migration. There are no admitted whole
softmax, logsumexp, normalization, fusion, or online-recurrence rules.

In particular, the table has no exp/log identity, no max identity and no fp64
instance. `SQRT-RSQRT` was not admitted. `DIV-RCP` was measured with Triton's
`div_rn`; an ordinary or approximate division cannot silently use that result.

The current `Spec.Derivation` supports atoms, symmetry, transitivity and common
sequential context. Additional proof infrastructure is needed for rewriting
inside expressions and loop bodies, reduction trees, and memory fusion. Such
infrastructure must preserve numerical operations; real ring identities cannot
be installed as structural FP rules.

Legacy `KernelIO.Equiv` proofs quantify over a boundary-rounding model. They
are not proofs under the new two-gates-selected atom calculus and do not count
as completed FP entries in this ledger.

For real correctness, explicit narrow-float storage annotations must also be
erased: merely setting rounding to the identity does not make a typed fp32 or
bf16 memory cell a real memory cell. This distinction matters for the
`KernelIO` output readback. FP proofs retain dtype annotations.

## Validation

The current migration passes 36 regression tests. The checks cover:

- `lake build TritonBenchSpecExamples` (all currently present example modules).
- The admission, assumption-printer and specification-surface tests. Repeated
  uses of the same admission at different rewrite sites print once; distinct
  contracts or evidence remain distinct, even if their names or keys coincide.
- Pair checks: the real/FP originals have equal mathematical projections, FP
  files do not import Correct, and real float addition includes empty tiles.
- Exact source equality against both original kernels in each of the four
  reduction pairs, plus their public mathematical formulas. The real proofs
  include output readback, termination, bounds safety and memory framing.
- Official comparator replay with trust audits for `VectorAddFPEquiv`,
  `FlatVectorAddFPEquiv`, `FloatDTypeAddFPEquiv`, `FloatDTypeAddCorrect`,
  `SoftmaxStableCorrect`, `StableLogSumExpCorrect`, `SoftmaxReciprocalCorrect`,
  `FloatDTypeSoftmaxCorrect`, `HyperConnectionsDepthFPEquiv`,
  `HyperConnectionsWidthFPEquiv` and `AdamUpdateGridLaunchFPEquiv`.

Compile each changed module and run its axiom/statement audits. The
`TritonBenchSpecExamples` Lake target now includes all `bench.examples` modules,
so missing companion dependencies cannot hide behind the lightweight default
library build. Run the admission/printer tests after changing their consumers,
and run the independent comparator on the resulting proofs before declaring
the migration complete.
