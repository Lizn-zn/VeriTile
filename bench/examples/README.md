# Worked examples

Each example has its own directory. Start with `Kernels.lean` to read the
Triton programs, then choose the specification you want to inspect:

| File | Purpose |
| --- | --- |
| `Kernels.lean` | Original, optimized and intermediate source programs, including precision, casts, masks and stores. |
| `Correct.lean` | Real mathematical correctness of both original and optimized implementations against the stated formula, using their real projections. |
| `FPEquiv.lean` | FP equivalence under the selected atomic assumptions, with `#print_fp_assumptions` on completed specifications. |
| `Execution.lean`, `Contract.lean`, `Comparison.lean`, `Batch.lean` | Example-specific execution and composition lemmas, when needed. |
| `RealEquiv.lean` | Retained proofs using real intermediate arithmetic and their stated cast semantics. These are separate from the two-gates FP specifications. |

`Correct.lean` and `FPEquiv.lean` reference the same typed source definitions
without importing one another. Names such as `originalKernel` only specialize
region names or the documented scalar slice; they do not duplicate the source.
Correctness uses the real projection of that source, erasing casts when required.
In `FloatDTypeSoftmax`, both proofs retain the separate `x32` load and `x` cast
assignments; only their numerical interpretation differs.

Experiment shape and distribution select atomic assumptions. They do not fix
the dimensions of the subsequent proofs. Precision, intrinsic choice and
required input/intermediate domains remain part of the FP statements.

| Example | Transformation and existing proof scope |
| --- | --- |
| [TritonBenchVectorAddition](TritonBenchVectorAddition/) | Masked TritonBench addition; FP addition commutation. |
| [VectorAdd](VectorAdd/) | Aligned addition; FP addition commutation. |
| [FlatVectorAdd](FlatVectorAdd/) | Masked addition; FP addition commutation. |
| [FloatDTypeAdd](FloatDTypeAdd/) | Explicitly typed addition; FP addition commutation. |
| [AdamUpdateGridLaunch](AdamUpdateGridLaunch/) | In-place Lion update (the source is named Adam); commute the update's addition. |
| [HyperConnectionsDepth](HyperConnectionsDepth/) | Symbolic `S/T/D` and normalization count; commute the final pointwise addition. General real matrix formula and contextual FP proof; scalar flat-memory proof retained. |
| [HyperConnectionsWidth](HyperConnectionsWidth/) | Symbolic `S/T/D` and normalization count; replace logit division by reciprocal multiplication, preserving both matrix products. General real formulas and contextual FP proof; scalar specialization retained. |
| [RowWiseMax](RowWiseMax/) | Inline the maximum computation; structural FP equivalence without numerical assumptions. |
| [RowWiseSum](RowWiseSum/) | Decompose observable outputs and prove reversed reduction lanes using guarded addition commutation, association and zero identities. |
| [FusedSiLU](FusedSiLU/) | Fuse the pipeline; structural FP equivalence without numerical assumptions. |
| [FusedSwiglu](FusedSwiglu/) | Fuse the masked pipeline while retaining casts; structural FP equivalence without numerical assumptions. |
| [SoftmaxReciprocal](SoftmaxReciprocal/) | Replace per-lane division with reciprocal multiplication; unchanged tl.exp prefix, only div_mul_rcp assumed. |
| [FloatDTypeSoftmax](FloatDTypeSoftmax/) | The reciprocal rewrite with fp32 loads/stores and fp64 work; unchanged tl.exp prefix, only the casted div_mul_rcp atom assumed. |
| [SoftmaxStable](SoftmaxStable/) | Naive versus max-shifted softmax, derived from scalar assumptions and reduction plans. |
| [OnlineSoftmax](OnlineSoftmax/) | Complete two-pass online kernel versus batch softmax; successful executions, stored output rows and memory frames. |
| [Welford](Welford/) | Two-pass versus online mean/variance, with the stated count bound and reduction plan. |
| [FusedLayerNorm](FusedLayerNorm/) | Two-pass versus Welford-based LayerNorm, with the stated count bound and domains. |
| [LogExp](LogExp/) | Eliminate the masked, piecewise libdevice log-exp expression using its admitted atom. |
| [StableLogSumExp](StableLogSumExp/) | Fallback-preserving optimized source versus the unchanged direct tl.log reference, proved using `log_mul_split .tl`, `log_exp_cancel .tl`, libdevice exp-sub and arithmetic. Symbolic row size, reduction schedule, bf16 output and frame retained. The historical unconditional target is isolated in `Unconditional.lean` and remains unadmitted. |

Both versions now have real correctness specifications for vector addition,
the Lion update, general mHC matrices and their scalar specializations, reversed row sum and inlined row max.
Each real statement includes successful execution and a memory frame. The new general mHC and complete online statements use region memory; the retained scalar/batch KernelIO statements additionally bridge to bounded flat memory. Empty tiles are covered by the optimized aligned/float/TritonBench adds,
row sum and Lion update; masked FlatVectorAdd and row max retain their stated
positive-block-size conditions. OnlineSoftmax proves actual stored outputs; the recurrence-only helper remains available separately.

The seven local-rewrite examples retain the finite operands of each actual
rewrite in `GuardedRewrite.Program.domain`. The checked contextual lemmas
preserve the whole execution result, including all registers and memory.
They preserve successful runs and also agree on failure; they do not claim
that every possible surrounding instruction can execute. Masked additions
require finite evaluated operands even on inactive lanes, since these statements
still compute the addition before the masked store.

RowWiseSum uses successful execution summaries and explicit reduction trees.
The domain lists the intermediate operands of both normalization paths; finite
input leaves alone are insufficient. The proof uses add_commute, add_assoc and
add_zero, including the zero steps introduced by normalization. Padding stays
explicit in the original schedules, and row length/stride remain symbolic.
The generic tactics remain available for syntax derivations and their existing
IO adapter; see [the tactic guide](../../documents/EquivalenceTactics.md).
These guarded examples use explicit domain-preserving composition lemmas.
The other completed FP examples connect their transformations
to execution or observable outputs. A proof of one stated slice or observation
does not certify a larger kernel or a different precision/intrinsic variant.

Build all example modules with `lake build TritonBenchSpecExamples`.
The Python regressions in `scripts/test_example_spec_pairs.py` and
`scripts/test_fp_*.py` check the source bindings, independent proof imports,
domain boundaries and printed assumptions.
