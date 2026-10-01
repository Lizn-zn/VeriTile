# Example specification migration

The reference is `TritonBenchVectorAdditionCorrect.lean` and
`TritonBenchVectorAdditionFPEquiv.lean`.

Each completed case has a real correctness file and an independent FP equivalence file.
The correctness statement is `Spec.Real (io ⊨ mathematical_formula)`.
The FP statement is `originalKernel … ≡[R] optimizedKernel …` (or the corresponding
IO contracts for transformations with private scratch), with
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
| Row-wise sum | `RowWiseSumCorrect` — checked | `RowWiseSumFPEquiv` — derived from fp32 add commutation and association; forward versus reversed input lanes, arbitrary stride and block size including zero |
| Row-wise max | `RowWiseMaxCorrect` — checked | `RowWiseMaxFPEquiv` — checked; inline the load and reduction into the store, preserving the same reduction and input order; no numerical assumptions |
| Online softmax | `OnlineSoftmaxCorrect` — checked, original batch-kernel/online-recurrence scope | Batch versus online: pending elementary exp laws and loop/reduction derivation |
| mHC depth | `HyperConnectionsDepthCorrect` — checked, original rank-one/zero-iteration scope | `HyperConnectionsDepthFPEquiv` — checked in the same scope; add commutation |
| mHC width | `HyperConnectionsWidthCorrect` — checked, original rank-one/zero-iteration scope | `HyperConnectionsWidthFPEquiv` — checked in the same scope; two multiplication commutations |
| Adam-named Lion update | `AdamUpdateGridLaunchCorrect` — checked, per-program and grid proofs retained | `AdamUpdateGridLaunchFPEquiv` — checked per program; momentum addition commutation, masked in-place stores retained |
| Stable softmax | `SoftmaxStableCorrect` — checked for both original kernels against the softmax formula | Naive versus stable: missing admitted elementary exp laws |
| Stable logsumexp | `StableLogSumExpCorrect` — checked for both original kernels against logsumexp | Direct versus stable: missing admitted elementary exp/log laws |
| Softmax reciprocal | `SoftmaxReciprocalCorrect` — checked for both original kernels against the softmax formula | Division versus reciprocal multiplication: pending a faithful match to tested `div_rn` |
| Float dtype softmax | `FloatDTypeSoftmaxCorrect` — checked for both original fp32-load/fp64-work kernels against the softmax formula | Original fp64 intermediate arithmetic has no admitted fp64 row |
| Fused SiLU | `FusedSiLUCorrect` — checked for both original kernels against residual + silu(x · gate), including empty blocks and scratch framing | `FusedSiLUFPEquiv` — checked; original fused versus materialized pipeline, with no numerical assumptions |
| Fused SwiGLU | `FusedSwigluCorrect` — checked for both original kernels against silu(x) · y, including empty blocks, tail masks and scratch framing | `FusedSwigluFPEquiv` — checked; original fused versus materialized pipeline, with bf16 casts, tail masks and no numerical assumptions |
| Welford | `WelfordCorrect` — checked for both original kernels against population mean and variance; both output windows and memory framing | Two-pass versus online variance: missing initialization/division relations, then loop/reduction derivation |
| Fused layernorm | `FusedLayerNormCorrect` — checked for both original kernels against population-variance normalization and affine transformation | Two-pass versus online statistics: missing scalar relations for statistics, then loop/reduction derivation |

There are currently 18 correctness modules and 11 FP equivalence modules. The
eight legacy equivalence modules remain as source references; six of their
original transformations still await FP migration. Their presence does not
complete the pending FP entries above.

The SiLU and SwiGLU materialized kernels retain the original `ComputeKernel.seq`
scope: one concatenation of stage bodies. Both real and FP specifications explicitly
declare scratch windows and prove that every cell outside output and scratch
windows is preserved. They do not claim a new separate-launch theorem.

Welford and LayerNorm retain symbolic row length and stride. These are
per-program specifications: the original Welford sources write each scalar
output at offset zero, so this does not assert a race-free multi-program
Welford launch. Both proofs also cover zero-length rows under Lean's total
real arithmetic. For Welford, that means zero mean and variance; for LayerNorm,
there are no output lanes. The new `KernelIO₁ₓ₂.Implements` interface requires
both results and frames outside the union of the two output windows and any
declared scratch windows.

## Admission and proof boundaries

The trusted PR #9 report admits 30 instances of elementary relations. Its
accepted table is unchanged by this migration. There are no admitted whole
softmax, logsumexp, normalization, fusion, or online-recurrence rules.

In particular, the table has no exp/log identity, no max identity and no fp64
instance. `SQRT-RSQRT` was not admitted. `DIV-RCP` was measured with Triton's
`div_rn`; an ordinary or approximate division cannot silently use that result.

[The remaining prerequisites](./FPRemainingAdmissionGaps.md) are backed by
Lean-checked scalar countermodels satisfying all currently admitted rule
families. In particular, the table does not force additive identity, Welford
initialization, ordinary-division replacement or the exp/log transformations.
These are algebraic coverage checks, not numerical gate rejections. The
remaining entries cannot be completed merely by adding more proof automation.

The current `Spec.Derivation` supports atoms, symmetry, transitivity and common
sequential context. A `ProgramSyntax` view may additionally enable independently
proved structural execution steps through `Spec.ProgramDerivation`, without
changing the public `≡[R]` notation or the existing syntax-only views. Each step
preserves the public signature; syntax steps also preserve private scratch
metadata. Whole-program structural steps cannot be framed inside arbitrary
statement contexts.

`Float/Structural` interprets floating values with an arbitrary carrier and
arbitrary numerical functions, preserving dtype and compute-precision tags.
`Float/StructuralIO` requires both executions to succeed, output cells to agree,
and every cell outside each implementation's output and private scratch windows
to remain unchanged. Scratch cannot alias public inputs or output. The SiLU
proof retains the exact original kernels and derives the same opaque numerical
call tree by store/load forwarding. Its assumption printer reports `none`.
This is a structural theorem of the FP model, not an IEEE execution theorem or
a claim that a newly measured whole-kernel numerical test passed.

The SwiGLU FP proof uses the actual elaborated syntax: the intermediate load
annotation becomes a bf16-typed load, with no additional cast. Its legacy
proof needed rounding idempotence because that model rounded again on a typed
store. The structural model copies an already typed value on store and retains
all explicit casts as opaque operations. The new proof therefore also reports
`none`. Its `MaskedKernelIO₂` signature preserves the complete active-lane
predicate, and its frame preserves inactive output and scratch cells. It works
for arbitrary element count and block size, including zero and partial blocks;
inactive scratch values are not used as a forwarding premise.

The structural evaluator currently supports straight-line assignments, typed
loads/stores, masks and a subset of expressions. Max reduction is interpreted
as an arbitrary operation retaining compute precision, the entire input tile,
shape, axis and `keepDims`; an empty reduction axis fails, as in the original
semantics. The row-wise max FP proof preserves that operation while eliminating
the `values` and `result` register bindings. Its stride and positive row length
are symbolic, with the same nonempty-row condition as its Correct counterpart.
Its one-input IO interface requires successful executions, equal output cells
and memory framing on each side.

The row-wise sum proof specializes the original mathematical kernel to fp32
input and accumulation and reverses the lane addresses before `tl.sum`. The
DSL preserves the input's fp32 compute annotation on the reduction and its
result. The mathematical projection still equals `RowWiseSumCorrect`'s kernel.

`Float/Equational` supplies expression congruence and substitution of the exact
scalar ADD-COMMUTE and ADD-ASSOC templates. Its reduction-tree theorem derives
equivalence from a permutation of the leaves, without a whole-reduction atom
or a floating additive-identity assumption. `Float/TermModel` expands sum into
an arbitrary valid addition schedule. Each input occurs exactly once; any
explicit padding zeros remain leaves with their multiplicity preserved. The
schedule retains precision and layout and cannot depend on numerical inputs.
A concrete valid schedule exists even for empty rows.

The one-input IO view can now use these term derivations to relate actual
successful abstract executions, with the same typed-output and memory-frame
obligations as its structural steps. Other primitives stay opaque. The public
notation remains `lhs ≡[R] rhs`, and the sum example prints only `add_assoc`
and `add_commute`. This derives a theorem in the selected FP model; it neither
replays the GPU report nor claims an IEEE or whole-kernel statistical guarantee.

Unsupported syntax still fails explicitly. Further infrastructure is needed
for loop bodies and additional memory transformations. Real ring identities
cannot be installed as structural FP rules.

Legacy `KernelIO.Equiv` proofs quantify over a boundary-rounding model. They
are not proofs under the new two-gates-selected atom calculus and do not count
as completed FP entries in this ledger.

For real correctness, explicit narrow-float storage annotations must also be
erased: merely setting rounding to the identity does not make a typed fp32 or
bf16 memory cell a real memory cell. This distinction matters for the
`KernelIO` output readback. FP proofs retain dtype annotations.

## Validation

The checks cover:

- `lake build TritonBenchSpecExamples` (all currently present example modules).
- The admission, assumption-printer and specification-surface tests. Repeated
  uses of the same admission at different rewrite sites print once; distinct
  contracts or evidence remain distinct, even if their names or keys coincide.
- Pair checks: the real/FP originals have equal mathematical projections, FP
  files do not import Correct, and real float addition includes empty tiles.
- Exact source equality against both original kernels in each of the four
  reduction pairs, plus their public mathematical formulas. The real proofs
  include output readback, termination, bounds safety and memory framing.
- Exact source equality for both SiLU and SwiGLU pairs, and applications of all
  four real correctness headlines at arbitrary dimensions, without positivity
  or whole-tile restrictions.
- Exact source equality for both kernels in `FusedSiLUFPEquiv`, its public FP
  theorem at symbolic block size (including zero), independence from Correct,
  and its empty numerical-assumption output. Structural countermodels prevent
  accidental commutation, reassociation, cast idempotence or precision erasure;
  they also check typed forwarding, register shadowing, unsupported executions,
  private scratch and cell-level framing.
- Exact source equality for the original fused and materialized SwiGLU kernels,
  its independent FP proof at symbolic dimensions, and its empty assumption
  output. Masked-interface counterexamples prevent changing the public mask,
  hiding writes to inactive output or scratch lanes, aliasing input as scratch,
  or treating two failed executions as a structural certificate.
- Exact source equality for row-wise max, independent FP proof and empty
  assumption output, and application of its public theorem at symbolic stride
  and positive block size. Countermodels keep max input permutations distinct
  and reject empty max reductions; the one-input interface also protects output
  windows, private scratch and successful-execution requirements.
- Row-wise sum's original mathematical projection, its independent FP proof at
  arbitrary dimensions including empty rows, and exact two-atom output.
  Reduction-tree countermodels reject removing a zero leaf, erasing precision
  or dropping repeated casts. Opaque sum interpretation does not permit input
  permutations; numerical IO steps still reject failed executions and dtype
  changes. Assumption auditing distinguishes internal induction hypotheses
  from opaque external derivation or execution premises.
- Exact source equality for both Welford and LayerNorm pairs, and applications
  of all four public real formulas without positivity restrictions. A flat
  memory consumer checks Welford's two numerical outputs and preservation of
  unwritten cells within both output regions.
- Official comparator replay with trust audits for `VectorAddFPEquiv`,
  `FlatVectorAddFPEquiv`, `FloatDTypeAddFPEquiv`, `FloatDTypeAddCorrect`,
  `SoftmaxStableCorrect`, `StableLogSumExpCorrect`, `SoftmaxReciprocalCorrect`,
  `FloatDTypeSoftmaxCorrect`, `HyperConnectionsDepthFPEquiv`,
  `HyperConnectionsWidthFPEquiv`, `AdamUpdateGridLaunchFPEquiv`,
  `FusedSiLUCorrect`, `FusedSwigluCorrect`, `WelfordCorrect`,
  `FusedLayerNormCorrect` and the updated `KernelSpec/Basic` interface. The
  structural FP extension also replays `Spec`, `Float/Structural`,
  `Float/StructuralIO`, `FusedSiLUFPEquiv`, `FusedSwigluFPEquiv`,
  `RowWiseMaxFPEquiv` and the boundary fixture, including its named reduction
  and IO-interface counterexamples.
  The sum extension replays `Float/Equational`, `Float/TermModel`,
  `RowWiseSumFPEquiv` and their updated interfaces and boundary fixtures;
  all 510 theorem targets in that batch were accepted.

The regression suite includes 7 example-pair/contract tests, 7 structural FP
tests, 6 expression/reduction/admission-coverage tests and 35 admission,
assumption-printer and specification-surface tests.

The comparator also has 7 passing integration tests. Its theorem inventory
enumerates and looks up declarations in the same completed kernel environment,
so realized private match equations remain explicit replay targets even when
the elaborator's name lookup cannot retrieve them. The tests retain rejection
checks for `sorry` and unapproved axioms.

Compile each changed module and run its axiom/statement audits. The
`TritonBenchSpecExamples` Lake target now includes all `bench.examples` modules,
so missing companion dependencies cannot hide behind the lightweight default
library build. Run the admission/printer tests after changing their consumers,
and run the independent comparator on the resulting proofs before declaring
the migration complete.
