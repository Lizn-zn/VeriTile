# rms_norm_triton

`Kernels.lean` contains the historical source declaration and the complete typed
original/candidate family. X and Y share a storage type; W independently ranges
over fp16, bf16, fp32 and fp64, giving 16 storage pairs. The source retains all
11 statements, symbolic row/column strides, masked loads with zero padding,
fp32 normalization and runtime epsilon, and the cast to Y's element type
**before** multiplication by W. Multiplication promotes its operands according
to their types, and the store converts to Y when required. Local precision scopes
preserve the inner fp32 operations under an enclosing fp64 operation.

The candidate commutes the two natural-number row-base products. This is an
exact address canonicalization; no speedup or floating arithmetic rewrite is
claimed.

`Correct.lean` preserves the legacy contract and proves both typed versions
against the independent finite-sum formula in `Proofs/Math.lean`. The new full-row
contracts require `0 < N`, `N ≤ B`, and a positive output column stride. They
quantify over every program row, active-lane bounds, disjoint flat placements,
and matching input values; they prove successful execution, every active
output, and the complete memory frame. Natural strides are nonnegative, so
negative strides are outside this interface. Real sqrt/division use Lean's
total real operations; no epsilon positivity is assumed. The legacy empty-tile
contract remains unchanged.

`FPEquiv.lean` proves complete typed-state and failure equivalence with an empty
floating-assumption list, under the default fp32 profile and explicit local
fp64 scopes. It preserves every register, memory cell, mask, reduction order,
cast and intrinsic. `Proofs/SourceChecks.lean` checks all 16 pairs retain the
pre-weight cast. `Proofs/Execution.lean` supplies successful runs of both full
programs using an exact-real `WithBot ℝ` backend: fp16 X, bf16 W, program row 1,
row strides 3/6 and column stride 2, one active lane and one masked tail,
X=3, W=2, eps=7, output 3/2, and untouched fp16 tail value 99. This is a
nonempty numerical execution witness, not an IEEE rounding certificate.

The Python wrapper's contiguous copies, allocation, launch grid, device choice,
and shape validation are trusted. The source test exercises fp32; the typed
proof family additionally covers the other stated storage pairs. Shared
regressions `bench/tests/LocalPrecisionDSL.lean` and
`bench/tests/LocalPrecisionScope.lean` check source AST precision boundaries,
actual backend callback precision, erasure, and failure preservation.

Mixed fp16/bf16 multiplication prefers fp16, following the
[Triton tensor promotion rule](https://triton-lang.org/main/python-api/triton-semantics.html).

The integrated public `Correct`/`FPEquiv` dependency build passes (3445 jobs), after the final promotion correction. Five contracts have validated source fingerprints and `test-gap` status; independent comparator replay remains pending.
