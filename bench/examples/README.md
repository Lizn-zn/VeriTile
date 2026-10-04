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
| [HyperConnectionsDepth](HyperConnectionsDepth/) | Depth connection, `S=T=D=1`, zero normalization iterations; commute the final addition. |
| [HyperConnectionsWidth](HyperConnectionsWidth/) | Width connection, the same scalar slice; commute the selected multiplications. |
| [RowWiseMax](RowWiseMax/) | Inline the maximum computation; structural FP equivalence without numerical assumptions. |
| [RowWiseSum](RowWiseSum/) | Reverse reduction lanes using admitted addition commutation and association. |
| [FusedSiLU](FusedSiLU/) | Fuse the pipeline; structural FP equivalence without numerical assumptions. |
| [FusedSwiglu](FusedSwiglu/) | Fuse the masked pipeline while retaining casts; structural FP equivalence without numerical assumptions. |
| [SoftmaxReciprocal](SoftmaxReciprocal/) | Replace per-lane division with reciprocal multiplication. |
| [FloatDTypeSoftmax](FloatDTypeSoftmax/) | The reciprocal rewrite with fp32 loads/stores and fp64 work. |
| [SoftmaxStable](SoftmaxStable/) | Naive versus max-shifted softmax, derived from scalar assumptions and reduction plans. |
| [OnlineSoftmax](OnlineSoftmax/) | Batch output versus normalization recovered from the online kernel's final `m/l` registers; the online source has no output store. |
| [Welford](Welford/) | Two-pass versus online mean/variance, with the stated count bound and reduction plan. |
| [FusedLayerNorm](FusedLayerNorm/) | Two-pass versus Welford-based LayerNorm, with the stated count bound and domains. |
| [LogExp](LogExp/) | Eliminate the masked, piecewise libdevice log-exp expression using its admitted atom. |
| [StableLogSumExp](StableLogSumExp/) | Both real correctness proofs complete. FP target pending `log_mul` and `log_exp_libdevice` admission; `FPEquiv.lean` records a goal, not a certificate. |

Both versions now have real correctness specifications for vector addition,
the Lion update, the fixed-rank mHC slices, reversed row sum and inlined row max.
Each statement includes successful execution, its IO bounds and the memory
frame. Empty tiles are covered by the optimized aligned/float/TritonBench adds,
row sum and Lion update; masked FlatVectorAdd and row max retain their stated
positive-block-size conditions. OnlineSoftmax retains the batch/recurrence
observation scope described above.

The seven scalar-rewrite examples at the top derive equivalence of typed
program bodies. The other completed FP examples connect their transformations
to execution or observable outputs. A proof of one stated slice or observation
does not certify a larger kernel or a different precision/intrinsic variant.

Build all example modules with `lake build TritonBenchSpecExamples`.
The Python regressions in `scripts/test_example_spec_pairs.py` and
`scripts/test_fp_*.py` check the source bindings, independent proof imports,
domain boundaries and printed assumptions.
