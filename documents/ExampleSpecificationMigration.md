# Example specification migration

The reference is `TritonBenchVectorAddition/Correct.lean` and
`TritonBenchVectorAddition/FPEquiv.lean`.

Each completed case has a real correctness file and an independent FP equivalence file.
The correctness statement is `Spec.Real (io ⊨ mathematical_formula)`.
The FP statement is `originalKernel … ≡[R] optimizedKernel …` (or the corresponding
IO contracts for transformations with private scratch), with
`#print_fp_assumptions` listing the numerical atoms used by its proof. Both proof files import the shared sources in `Kernels.lean`. FP execution
and contract helpers live in the same directory and never import the
correctness counterpart. See the [example index](../bench/examples/README.md)
for the source variants and precise observation scopes.

Experimental shape and distribution select assumptions. They are not extra
shape conditions on the subsequent Lean derivation. Kernel dimensions remain
parameters except where a case was already explicitly a fixed-rank slice.

## Coverage ledger

This table tracks the original cases, not just newly compiling files. A real
proof or an unrelated local rewrite does not complete an existing algorithmic
FP equivalence. Pending entries must not be advertised as proved.

| Original case | Correctness file | FP transformation and status |
|---|---|---|
| TritonBench vector addition | `TritonBenchVectorAddition/Correct.lean` — checked | `TritonBenchVectorAddition/FPEquiv.lean` — checked; add commutation |
| Aligned vector addition | `VectorAdd/Correct.lean` — checked | `VectorAdd/FPEquiv.lean` — checked; add commutation |
| Masked vector addition | `FlatVectorAdd/Correct.lean` — checked | `FlatVectorAdd/FPEquiv.lean` — checked; add commutation |
| Float dtype addition | `FloatDTypeAdd/Correct.lean` — checked, including empty tiles | `FloatDTypeAdd/FPEquiv.lean` — checked; add commutation; output cast retained |
| Row-wise sum | `RowWiseSum/Correct.lean` — checked | `RowWiseSum/FPEquiv.lean` — checked under the admitted fp32 ADD-COMMUTE and ADD-ASSOC assumptions; dimensions and reduction schedules remain symbolic |
| Row-wise max | `RowWiseMax/Correct.lean` — checked | `RowWiseMax/FPEquiv.lean` — checked; inline the load and reduction into the store, preserving the same reduction and input order; no numerical assumptions |
| Online softmax | `OnlineSoftmax/Correct.lean` — checked, original batch-kernel/online-recurrence scope | `OnlineSoftmax/FPEquiv.lean` — checked in that same observation scope: stored batch values versus read-only normalization of the actual online m/l registers; libdevice EXP-SUB and scalar arithmetic, symbolic positive row length and separate memory frames |
| mHC depth | `HyperConnectionsDepth/Correct.lean` — checked, original rank-one/zero-iteration scope | `HyperConnectionsDepth/FPEquiv.lean` — checked in the same scope; add commutation |
| mHC width | `HyperConnectionsWidth/Correct.lean` — checked, original rank-one/zero-iteration scope | `HyperConnectionsWidth/FPEquiv.lean` — checked in the same scope; two multiplication commutations |
| Adam-named Lion update | `AdamUpdateGridLaunch/Correct.lean` — checked, per-program and grid proofs retained | `AdamUpdateGridLaunch/FPEquiv.lean` — checked per program; momentum addition commutation, masked in-place stores retained |
| Stable softmax | `SoftmaxStable/Correct.lean` — checked for both original kernels against the softmax formula | `SoftmaxStable/FPEquiv.lean` — checked for the libdevice.exp kernels, using admitted scalar arithmetic and EXP-SUB; symbolic row length, scheduled sums, bf16 stores and frames retained |
| Piecewise log-exp | `LogExp/Correct.lean` — checked for the original and copy kernels against the identity formula | `LogExp/FPEquiv.lean` — checked using the admitted masked libdevice log-exp/expm1 atom |
| Stable logsumexp | `StableLogSumExp/Correct.lean` — checked for both original kernels against logsumexp | Direct versus stable libdevice kernels: EXP-SUB connected; fp32 LOG-MUL and `tl.log(libdevice.exp(a)) = a` still need admission |
| Softmax reciprocal | `SoftmaxReciprocal/Correct.lean` — checked for both original kernels against the softmax formula | `SoftmaxReciprocal/FPEquiv.lean` — ordinary fp32 division versus a shared reciprocal, with the original bf16 output cast and explicit finite/nonzero operand domain |
| Float dtype softmax | `FloatDTypeSoftmax/Correct.lean` — checked for both original fp32-load/fp64-work kernels against the softmax formula | `FloatDTypeSoftmax/FPEquiv.lean` — fp32 load, fp64 work, fp32 output; only the casted division/reciprocal relation is assumed |
| Fused SiLU | `FusedSiLU/Correct.lean` — checked for both original kernels against residual + silu(x · gate), including empty blocks and scratch framing | `FusedSiLU/FPEquiv.lean` — checked; original fused versus materialized pipeline, with no numerical assumptions |
| Fused SwiGLU | `FusedSwiglu/Correct.lean` — checked for both original kernels against silu(x) · y, including empty blocks, tail masks and scratch framing | `FusedSwiglu/FPEquiv.lean` — checked; original fused versus materialized pipeline, with bf16 casts, tail masks and no numerical assumptions |
| Welford | `Welford/Correct.lean` — checked for both original kernels against population mean and variance; both output windows and memory framing | `Welford/FPEquiv.lean` — checked under scalar arithmetic and the two bounded count atoms, for `0 < N <= 2^24`; both original bf16 outputs and memory frames retained |
| Fused layernorm | `FusedLayerNorm/Correct.lean` — checked for both original kernels against population-variance normalization and affine transformation | `FusedLayerNorm/FPEquiv.lean` — checked for `N <= 2^24`, including empty output rows; original statistics, affine suffix, bf16 stores and frames retained |

There are 19 correctness modules, 18 completed FP equivalence modules, and
one goal-only FP file for StableLogSumExp. The eight `RealEquiv.lean` modules
retain proofs with real intermediate arithmetic and their stated cast semantics.
Their presence does not complete a pending FP transformation.

The SiLU and SwiGLU materialized kernels retain the original `ComputeKernel.seq`
scope: one concatenation of stage bodies. Both real and FP specifications explicitly
declare scratch windows and prove that every cell outside output and scratch
windows is preserved. They do not claim a new separate-launch theorem.

Welford and LayerNorm retain symbolic row length and stride. The FP count rules additionally require `N <= 2^24`;
Welford requires a nonempty row because its empty mean divides by zero. These are
per-program specifications: the original Welford sources write each scalar
output at offset zero, so this does not assert a race-free multi-program
Welford launch. Both proofs also cover zero-length rows under Lean's total
real arithmetic. For Welford, that means zero mean and variance; for LayerNorm,
there are no output lanes. The new `KernelIO₁ₓ₂.Implements` interface requires
both results and frames outside the union of the two output windows and any
declared scratch windows.

## Admission and proof boundaries

The main, supplemental and bounded-count reports define the accepted precision
instances. They use a local-ULP mean-bias budget (`tau=0.05`, five SEs) and the
peak absolute-error ratio gate; z is diagnostic. Counts are computed from current
reports, and only dual-PASS rows enter the generated Lean tables. No table
contains a whole softmax, logsumexp, normalization, reduction or recurrence atom.

The supplemental EXP-SUB implementation uses libdevice.exp. Its bf16,
bf16-input/fp32-work/bf16-output and fp32 instances are admitted with the
configured magnitude PASS threshold of 10. Its identity is
part of the report contract. The configured tl.exp experiment rejected that intrinsic version
(B=0.1608954387 ULP > 0.05). The stable/online softmax and logsumexp examples
therefore explicitly use libdevice.exp in both their Correct and FP sources,
with separate AST and opaque FP symbols for the two implementations. LOG-MUL domain events and unsupported fp64 combinations
remain unaccepted. DIV-RCP uses div_rn; DIV-MUL-RCP tests ordinary division.

The two reciprocal examples use `Guarded.IO`: the signature includes a domain
contract checking finite exponential values and a finite, nonzero denominator
at the shared prefix. They quantify over opaque numerical interpretations
satisfying the selected scalar theory, prove both runs succeed, and frame every
cell outside the output window. Exp, max and sum are identical opaque operations
on both sides; no whole-softmax law or positivity of abstract exp is assumed.
Only the fp64 relation's final fp32 outputs are equated. The fp32 example lifts
its admitted fp32 relation through the common bf16 output cast.

`ComputeDType.fp64` and the DSL preserve explicit float64 casts and precision
through arithmetic, max, sum, exp and log in these examples. The wide FP example
binds its fp32 load before widening so each precision boundary is explicit.
Binary64 constant projection is partial (finite normals); the structural
interpreter deliberately rejects raw fp64 payload constants and typed fp64
loads, which these examples do not use. This is not a complete IEEE evaluator.

[The remaining prerequisites](./FPRemainingAdmissionGaps.md) distinguish the
main-table algebraic countermodels from the supplemental rule set. Stable
logsumexp remains pending. Its two scalar probes and paired libdevice.log
variants in `experiments/floating_point/supplement/log_accuracy_config.py`
completed on H200: LOG-MUL is bias-INCONCLUSIVE and LOG-EXP-LIBDEVICE is
bias-REJECT. Both variants have identical observations and PTX after removing
source-location directives under the recorded Triton 3.7.1 configuration.
None supplies the missing admission. The same profile also tests a new fp32
piecewise expression: log1p(expm1(a)) for abs(a)<=0.5, otherwise log(exp(a)).
It passes with B=0.0494428110 and U=0, but changes the operation and cannot
justify the original log-exp premise. See the [current log report](../experiments/floating_point/supplement/log_report/summary.md).

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
result. The mathematical projection still equals `RowWiseSum/Correct.lean`'s kernel.

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
notation remains `lhs ≡[R] rhs`. The sum example prints `add_commute`
and `add_assoc`, each bound to an accepted fp32 instance. This derives a theorem in the selected FP model; it neither
replays the GPU report nor claims an IEEE or whole-kernel statistical guarantee.

Unsupported syntax still fails explicitly. Counted loops and conditionals
now have execution lemmas. Welford also has scalar-derived loop and schedule
comparisons plus a scheduled IO contract with syntactic domain checks; its two
integer-count conversion atoms are bound to the accepted PR #12 report. The
public Welford FP theorem retains `0 < N <= 2^24`. LayerNorm reuses the
unrounded statistics through its unchanged affine suffix and completes its
three-input FP contract for `N <= 2^24`, including empty output rows. Real ring identities
cannot be installed as structural FP rules.

Stable softmax has independent libdevice source definitions and a scheduled
one-input contract. `SoftmaxStableFPEquiv.softmax_stable_equiv` derives its
reduction and normalization from the admitted scalar arithmetic and libdevice
EXP-SUB atoms. Max and bf16 casts remain opaque; the row length is symbolic
and positive. `#print_fp_assumptions` lists only the scalar assumptions.

`OnlineSoftmaxFPEquiv.online_softmax_equiv` compares `batchOutput` with
`normalizedOnline` using `Float/ObservedRow`. The latter executes the original
online kernel and then reads `exp(x - m) / l` using its actual final registers.
Both executions and all readbacks must succeed; missing registers cannot make
two failed observations count as equal. The batch retains its real-typed row
store and frames all other cells, while the online kernel preserves all memory.
The observation adds no store. Its signature retains the row layout, fp32
execution profile, symbolic size and reified finite/nonzero domain; syntax-only
steps cannot change the readback or memory frame. This preserves the existing
Correct scope, not an equality between the kernels' final memories.

Stable logsumexp retains its single bf16 store at `pid` while replacing both
exp implementations with libdevice.exp. Its exp-sub law is admitted; LOG-MUL
and `tl.log(libdevice.exp(a)) = a` remain explicit obligations. The existing
LOG-EXP experiment used tl.exp, so it does not cover the new composition.
Pending log and integer-conversion premises remain visible to the assumption
printer, including equations accessed through record fields.

Legacy `KernelIO.Equiv` proofs quantify over a boundary-rounding model. They
are not proofs under the new two-gates-selected atom calculus and do not count
as completed FP entries in this ledger.

For real correctness, explicit narrow-float storage annotations must also be
erased: merely setting rounding to the identity does not make a typed fp32 or
bf16 memory cell a real memory cell. This distinction matters for the
`KernelIO` output readback. FP proofs retain dtype annotations.

## Validation

The checks cover:

- The OnlineSoftmax observation update passes the full library/example build
  and independent comparator checking of 43 theorem targets across its view,
  execution contract, public specification and boundary fixture. Regressions
  check its eleven printed scalar atoms, source independence, missing-register
  failure, aliased readback and both memory frames. The paired log profile's
  ten fp32 kernel specializations compile for sm_90. All five cases completed
  4096 H200 replicates with identical independent CPU replay; LOG-MUL and its
  libdevice.log variant remain bias-INCONCLUSIVE, while LOG-EXP-LIBDEVICE and
  its libdevice.log variant are bias-REJECT. The added piecewise expm1/log1p
  expression passes, but does not admit the original operation. Fifteen GPU
  boundary inputs check small values, branch neighbors, fallback and overflow.
  CPU tests retain negative log-exp inputs and reject
  nonfinite outputs on valid inputs, and existing admission tables remain
  byte-for-byte reproducible from their frozen reports.
- The bounded-count update passes `lake build VeriTile TritonBenchSpecExamples`
  and 28 regression tests. It checks the two public FP specifications, their
  13 printed scalar atoms, source independence, the last admitted successor,
  a failing out-of-range conversion, and the common guarded result outside
  that range. Report-import tests reject incompatible scopes, precisions,
  incomplete/rejected results and contradictory metadata. LayerNorm's empty
  output and the admitted upper row-length boundary are exercised explicitly.
  Independent comparator replay accepts all 58 theorem targets in the count
  admission/binding, both new FP files, their boundary fixture and StatementAudit.
- The libdevice update passes `lake build VeriTile TritonBenchSpecExamples`
  and 26 related regression tests. The new fixture distinguishes the two
  exp symbols through nested syntax and fp32 annotations, gives them the
  same real meaning, and checks a model where libdevice EXP-SUB holds but
  intrinsic EXP-SUB does not. The printer tests cover derivation-table
  extension without hiding external symbolic atoms or opaque premises.
  Independent comparator replay accepts all 63 theorem targets across
  `StatementAudit`, `Float/Exponential`, `SoftmaxStable/FPEquiv.lean`, the online
  and logsumexp contracts, and `FPLibdeviceExp`.
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
- Exact source equality for both kernels in `FusedSiLU/FPEquiv.lean`, its public FP
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
- Row-wise sum's original mathematical projection, its FP proof under the
  admitted scalar assumptions at arbitrary dimensions including empty rows,
  and output identifying only add_commute and add_assoc.
  Reduction-tree countermodels reject removing a zero leaf, erasing precision
  or dropping repeated casts. Opaque sum interpretation does not permit input
  permutations; numerical IO steps still reject failed executions and dtype
  changes. Assumption auditing distinguishes internal induction hypotheses
  from opaque external derivation or execution premises.
- Exact source equality for both Welford and LayerNorm pairs, and applications
  of all four public real formulas without positivity restrictions. A flat
  memory consumer checks Welford's two numerical outputs and preservation of
  unwritten cells within both output regions.
- Official comparator replay with trust audits for `VectorAdd/FPEquiv.lean`,
  `FlatVectorAdd/FPEquiv.lean`, `FloatDTypeAdd/FPEquiv.lean`, `FloatDTypeAdd/Correct.lean`,
  `SoftmaxStable/Correct.lean`, `StableLogSumExp/Correct.lean`, `SoftmaxReciprocal/Correct.lean`,
  `FloatDTypeSoftmax/Correct.lean`, `HyperConnectionsDepth/FPEquiv.lean`,
  `HyperConnectionsWidth/FPEquiv.lean`, `AdamUpdateGridLaunch/FPEquiv.lean`,
  `FusedSiLU/Correct.lean`, `FusedSwiglu/Correct.lean`, `Welford/Correct.lean`,
  `FusedLayerNorm/Correct.lean` and the updated `KernelSpec/Basic` interface. The
  structural FP extension also replays `Spec`, `Float/Structural`,
  `Float/StructuralIO`, `FusedSiLU/FPEquiv.lean`, `FusedSwiglu/FPEquiv.lean`,
  `RowWiseMax/FPEquiv.lean` and the boundary fixture, including its named reduction
  and IO-interface counterexamples.
  The sum extension replays `Float/Equational`, `Float/TermModel`,
  `RowWiseSum/FPEquiv.lean` and their updated interfaces and boundary fixtures;
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
`TritonBenchSpecExamples` Lake target includes all `bench.examples` modules
and the imported TritonBench vector-addition module in its build globs,
so missing companion dependencies cannot hide behind the lightweight default
library build. Run the admission/printer tests after changing their consumers,
and run the independent comparator on the resulting proofs before declaring
the migration complete.
