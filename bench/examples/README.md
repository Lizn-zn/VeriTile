# Worked examples

Each example has its own directory. Start with `Kernels.lean` to read the
Triton programs, then choose the specification you want to inspect:

| File | Purpose |
| --- | --- |
| `Kernels.lean` | Original, optimized and intermediate source programs, including precision, casts, masks and stores. |
| `Correct.lean` | Public IO, necessary conditions, independent mathematical formulas and real correctness specifications, with short composition proofs. |
| `FPEquiv.lean` | Public IO, precision, numerical domains and FP equivalence specifications, with short composition proofs and `#print_fp_assumptions`. |
| `Proofs/Real.lean` | Supporting real execution and memory-bound lemmas, when needed. |
| `Proofs/FP.lean` | Supporting FP execution, operand-domain and composition lemmas, when needed. |
| `Proofs/LegacyCorrect.lean` | Real correctness of StableLogSumExp's historical unconditional source; the current pair stays in `Correct.lean`. |
| `Proofs/LegacyRealEquiv.lean` | Retained proofs using real intermediate arithmetic and their stated cast semantics. These are separate from the two-gates FP specifications. |
| `Proofs/Unconditional.lean` | Lean-checked disproof of StableLogSumExp's historical FP target without fallbacks; `UnconditionalModel.lean` supplies the logical countermodel. |

Only `Kernels.lean`, `Correct.lean` and `FPEquiv.lean` live at an example's
top level. Read these three files first; `Proofs/` contains supporting detail
and is omitted when no separate helpers are needed. Projection, execution,
memory bounds and real arithmetic belong in `Proofs/Real.lean`; FP execution,
recurrences and scalar derivations belong in `Proofs/FP.lean`.
Helper modules do not import reader entries, and real and FP layers remain
independent. Cross-example proof reuse imports the corresponding helper module.

IO windows, masks, scratch regions, precision profiles and numerical domains
remain visible in the entry files: these determine the statement itself.
Scheduled examples explicitly assemble their public contracts there. Lean checks
that those contracts match the internal execution interfaces used by the proofs.

Reduction examples use `FP.Kernel₁`, `FP.Kernel₁ₓ₂` or `FP.Kernel₃` from
`VeriTile.Triton.Float.Kernel`, according to the number of input/output regions.
Declare the reference with named `io`, `profile` and `domain` fields, then reuse
that configuration for the other source. `FP.Kernel₁` also lists `comparisons`:
`[]` means no comparison support is required; StableLogSumExp lists the two
operations used by its fallback. `FP.Kernel₁.ofExecution` reuses existing
reduction proofs, preserving their execution and memory-frame obligations.
`FP.fp32` names the default precision and sum-tree profile, preserving source casts.

LogExp and the reciprocal examples use `FP.PrefixKernel₁`: their domain checks
the actual registers after an explicit prefix, with `defaultPrecision` stated
in the entry. Pure structural proofs retain `KernelIO…`; local rewrites retain
their `Program` and its named `kernel`/`domain` fields. Their broader execution
scope is preserved. For IO contracts, `withKernel` replaces the source while
Lean checks its projection automatically; changes to scratch or masks remain
explicit. These helpers do not establish numerical equivalence.

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
| [HyperConnectionsDepth](HyperConnectionsDepth/) | Symbolic `S/T/D` and normalization count; commute the final pointwise addition. General real matrix formula with bounded flat memory; contextual FP proof uses the two tactics. |
| [HyperConnectionsWidth](HyperConnectionsWidth/) | Symbolic `S/T/D` and normalization count; replace logit division by reciprocal multiplication, preserving both matrix products. General real formulas with bounded flat memory; the two tactics compose both FP rewrite sites. |
| [RowWiseMax](RowWiseMax/) | Inline the maximum computation; structural FP equivalence without numerical assumptions. |
| [RowWiseSum](RowWiseSum/) | Decompose observable outputs and prove reversed reduction lanes using guarded addition commutation, association and zero identities. |
| [FusedSiLU](FusedSiLU/) | Fuse the pipeline; structural FP equivalence without numerical assumptions. |
| [FusedSwiglu](FusedSwiglu/) | Fuse the masked pipeline while retaining casts; structural FP equivalence without numerical assumptions. |
| [SoftmaxReciprocal](SoftmaxReciprocal/) | Replace per-lane division with reciprocal multiplication; unchanged tl.exp prefix, only div_mul_rcp assumed. |
| [FloatDTypeSoftmax](FloatDTypeSoftmax/) | The reciprocal rewrite with fp32 loads/stores and fp64 work; unchanged tl.exp prefix, only the casted div_mul_rcp atom assumed. |
| [SoftmaxStable](SoftmaxStable/) | Naive versus max-shifted softmax, derived from scalar assumptions and reduction plans. |
| [OnlineSoftmax](OnlineSoftmax/) | Complete two-pass online kernel versus batch softmax; successful executions, stored output rows and memory frames; real correctness includes bounded flat memory. |
| [Welford](Welford/) | Two-pass versus online mean/variance, with the stated count bound and reduction plan. |
| [FusedLayerNorm](FusedLayerNorm/) | Two-pass versus Welford-based LayerNorm, with the stated count bound and domains. |
| [LogExp](LogExp/) | Eliminate the masked, piecewise libdevice log-exp expression using its admitted atom. |
| [StableLogSumExp](StableLogSumExp/) | Fallback-preserving optimized source versus the unchanged direct tl.log reference, proved using `log_mul_split .tl`, `log_exp_cancel .tl`, libdevice exp-sub and arithmetic. Symbolic row size, reduction schedule, bf16 output and frame retained. `Proofs/Unconditional.lean` refutes the historical general FP target: a singleton logical countermodel satisfies all selected rules but stores different results. This does not assert a GPU or two-gates failure. |

Both versions now have real correctness specifications for vector addition,
the Lion update, general mHC matrices and their scalar specializations, reversed row sum and inlined row max.
Each real statement includes successful execution and a memory frame. General mHC and complete online correctness include bounded flat memory: buffer placements are disjoint, both input/output windows are bounded, and all other flat cells are preserved. Empty tiles are covered by the optimized aligned/float/TritonBench adds,
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
Both scalar and matrix local rewrites use `equiv_decompose` and `fp_prove`.
The `GuardedRewrite` and `ProfiledRewrite` adapters retain each actual execution
prefix and compose checked contextual lemmas. Domains locate assignments with
`Source.prefixBefore`, using a zero-based occurrence for repeated writes to the
same register. The proof checks that the selected prefix matches the source;
inserting an unrelated common statement does not require new statement offsets.
The other completed FP examples connect their transformations
to execution or observable outputs. A proof of one stated slice or observation
does not certify a larger kernel or a different precision/intrinsic variant.

Build all example modules with `lake build TritonBenchSpecExamples`.
The Python regressions in `scripts/test_example_spec_pairs.py` and
`scripts/test_fp_*.py` check the source bindings, independent proof imports,
domain boundaries and printed assumptions.
