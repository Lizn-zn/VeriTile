# fast_rms_layernorm

`Kernels.lean` contains the three complete JIT sources and their candidates:
regular RMSNorm forward, Gemma forward, and backward with both GEMMA constexpr
values. Each candidate only commutes the natural-number X row/stride product.
Input/output and weight storage independently range over fp16, bf16 and fp32.

Regular forward converts X to fp32, computes the reciprocal RMS, converts the
normalized value to the weight dtype, and multiplies in that precision. The
unique `W_row.dtype` is explicitly resolved from its direct typed W load.
Precision checks verify the conversion and multiplication for all nine storage
pairs. Gemma forward/backward instead convert their loads to fp32; final Y/dY
stores convert to input storage and the saved reciprocal RMS uses fp32.

`Correct.lean` exposes six independent real specifications for original and
candidate sources. They include successful finite flat execution, actual access
bounds, every active output and preservation of all other tagged memory cells.
Forward computes sum of squared active inputs divided by N, reciprocal RMS and
weighted output. Backward implements the source gradient formula using the
caller-supplied reciprocal RMS. The real contracts require nonempty rows
covered by the block; forward also requires distinct output regions and
positive mean square plus epsilon. The flat adapter requires finite disjoint
allocations, coverage and zero undefined values.

Source addressing differences are explicit. Regular forward reads W[j*WR];
Gemma forward and backward read W[j] even though the signature supplies WR.
Backward neither reads nor writes dW. Concrete finite witnesses use nonzero
row/statistic strides, an unallocated masked input tail and an integer-tagged
output neighbor. With WR=2, regular forward uses weights 1,3,5, while Gemma
forward uses contiguous weights 1,2,3 and adds one. Backward uses only three
weight cells even with WR=99 and no dW allocation; its two GEMMA branches have
different, checked gradients.

`FPEquiv.lean` exposes three complete typed-state/failure equivalences, with
empty domains and no numerical FP assumptions. All 36 original typed
configurations have candidate execution witnesses from empty registers. The
execution algebra demonstrates consistency, not physical GPU rounding. No
host launch, race, fixed-width overflow, calculus derivative or host
forward/backward composition claim is made.

Status: the complete source, real-contract and typed-equivalence review passed. Repository builds, source/dtype/layout gates and independent comparator/trust replay passed. No GPU experiment was run in this review.


Validation passed: complete source statement-tree comparison, repository proof
closure, precision and public finite/typed execution witnesses, source/layout
gates and 46 configured audit/tool/layout tests. The three-case independent
comparator/trust replay accepts all 52 selected source/proof/test files, with
current fingerprints and named theorem inventories. Receipt: `Logs/comparator-check-pq1sfkt8`.
This case's source/contract/equivalence review is **complete**, within the
explicit per-program mathematical/typed model and source behaviors above.
