# softmax_flaggems

**Complete symbolic per-program contracts on the stated source domains.**
`Correct.lean` proves all four complete kernels, both branch choices and both
versions. Five historical one-tile specifications retain their exact statements
and documentation in `Correct.lean`, with their source definitions in `Kernels.lean`
and helpers in `Proofs/LegacyReal.lean`. They remain separate from the four new
complete-source contracts. Historical scope notes are preserved in `LegacyReal.lean`
as documentation of those narrower declarations.

`Kernels.lean` now transcribes all four forward/backward kernels, both
`ONE_TILE_PER_CTA` branches, forward online recurrences, reverse output loops,
pointer rebinding and backward offset advances. Forward storage declarations
include independent fp16/bf16/fp32/fp64 input and output choices; backward uses
a common storage type. Unused Python `_` loop indices have a Lean identifier.
Explicit output casts represent the source store conversion. Launch heuristics,
wrapper composition and fixed-width address overflow remain outside the
per-program scope. Meaningful source launches require positive reduction length and tile dimensions
and a branch choice consistent with the source heuristics.

The candidate commutes the scalar Nat factors in program-base address
calculations. Inner forward changes its fallback pointer bases; its one-tile
branch is retained. Every floating operation, reduction schedule, loop and
padding expression is shared. `FPEquiv.lean` compares the typed lowered programs
with no numerical atoms; mathematical successful execution is a separate target.
This structural relation alone does not establish source-lowering fidelity.

`Proofs/RealProjection.lean` checks that all storage choices have the same erased
real program. `Proofs/RealCandidate.lean` proves equality of the complete original
and candidate real executions after arbitrary memory flattening, for all dimensions,
branch choices and initial states, including failure. This transports any later
original IO theorem to the candidate; it does not supply that IO theorem itself.

Pinned [Triton v3.5.1 standard.py](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/standard.py#L162-L176)
promotes bf16, and fp16 without returned indices, to fp32 in `tl.max`.
`Proofs/Source.lean` checks the exact promoted max, fp32 subtraction and fp32
exp operations, including precision annotations. The existing fp32 reduction
constructor is checked separately. The underlying typed-max AST constructor is
not the lowering of ordinary half/bf16 `tl.max`. Mixed fp32/half `tl.maximum`
promotion and pointer-rebinding load channels also have exact AST checks.
The fallback fp64 source profiles are unsupported: the forward loops initialize
`m`/`z` as fp32 and the backward loops initialize `scale` as fp32, but fp64 input
promotes their updates to fp64. Pinned Triton [promotion rules](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/semantic.py#L60-L95)
and its [loop-carried type check](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/compiler/code_generator.py#L963-L971)
imply rejection of those original source specializations. This is a source-code
inference, not an observed GPU compilation result. `Proofs/SourceSupport.lean`
records the explicit source-profile domain: one tile, or non-fp64 input.
The mathematical real projections and structural lowered-program equivalences
remain broader statements; they do not certify compilation of excluded profiles.
One-tile fp64 precision is checked in all four actual source branches, including
their loads, reductions and output arithmetic; pinned [tl.exp](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/math.py#L86-L92)
accepts fp64.

The source-facing forward fallback domain also requires `TILE_N ≤ N`. With
`N < TILE_N`, never-active lanes retain negative infinity and evaluate
`exp(-∞ - -∞)`, which can introduce NaNs into the denominator. The intended
wrapper fallback (`N > TILE_N`) satisfies this bound. `ForwardSourceDomain`
records it separately from dtype support. Broader ideal-real projection lemmas
use totalized `WithBot` arithmetic and are not claims about that IEEE boundary.

Backward masked loads omit `other`. Their undefined padding can contribute to
the reduction, so arbitrary padding does not imply the conventional softmax VJP.
`Proofs/RealMath.lean` defines the independent full-row softmax and VJP, gives the
exact padding contribution and proves the usual VJP when padded products vanish.
It does not assume zero padding implicitly. The source execution connections
use those formulas directly; concrete tests are not used as their correctness
premises.

`Proofs/RealMath.lean` also proves the scalar online denominator invariant,
masked-tail filtering, lane rescaling/collapse, and the tile/lane partition.
The backward lane-fold theorem and final reduction retain the entire padded row.
It constructs a zero-padding extension for arbitrary active backward data, so
the stated vanishing-padding-products condition is mathematically non-vacuous. `Proofs/RealAddress.lean`
proves reverse-tile permutation, exact ceiling loop counts and address arithmetic facts.
`Proofs/RealLoop.lean` proves successful dynamic-loop execution from an explicit
per-tile invariant and body transition, including a partial final tile. These reusable
lemmas are instantiated by the source-specific execution proofs below.

`Proofs/RealForwardNonInnerIO.lean` now connects the complete forward non-inner
two-pass branch (`ONE_TILE_PER_CTA = false`) to successful execution and the
independent full-column formula `exp(x_n) / sum_j exp(x_j)`. Both original and
candidate theorems cover every storage choice in the real projection, arbitrary
initial registers and memory, all active output coordinates, and preservation
of every full memory cell outside that program's output footprint. They also
preserve program identifiers, launch dimensions and undefined-padding values.
The premises are `0 < N`, `0 < TILE_N` and distinct input/output regions; they
contain no assumed execution result or output equation. The checked source
connection instantiates the online accumulation invariant, column reduction and
the actual reverse writeback loop. Axiom audits list only `propext`,
`Classical.choice` and `Quot.sound`.

`RealForwardNonInnerSafetyTransport.lean` proves full execution access safety
for both versions and all erased storage profiles when `pid_m < M` and both
region extents are at least `M*N*K`. `RealForwardNonInnerFlat.lean` lifts the
two-pass IO and full-cell frame to a disjoint flat allocation. This flat theorem
explicitly requires zero `undef`, as does the current `flattenState` bridge;
it does not replace or restrict the arbitrary-padding region theorem. The
combined `RealForwardNonInnerAll.lean` interface now also covers the one-tile
branch with `N ≤ TILE_N`, without an input/output alias restriction. Both branches
have checked successful execution, independent output formula, full region and
flat frames, and complete source access safety. Its axiom audit passes.

`RealBackwardNonInnerAll.lean` likewise covers both branches and all storage real
projections with successful region/flat IO, full frames and access safety. Its
formula retains actual masked-load padding; `RealBackwardNonInnerFormula.lean`
exposes the explicit vanishing-padding-products condition for the conventional
memory-only VJP. Both inner kernels now also have checked symbolic IO, full frames and source
access safety for both branches. `RealForwardInnerAll.lean` and
`RealBackwardInnerAll.lean` expose the same six original/candidate region IO,
access-safety and flat IO interfaces as the non-inner helpers. The forward inner
proof follows both read loops and both reverse write loops, including the
masked tail and unmasked full tiles.

The four public `Correctness.*_correctness` specifications combine both versions:
source access safety, successful region IO with arbitrary finite padding, and
successful full-frame flat IO conditional only on the explicit zero-undef bridge
premise. No execution or output equation is a proof premise. Source-profile and
forward fallback numerical domains are explicit in these new specifications;
all four dimension/alias domains have checked nonempty examples. The separate
real-model helper theorems deliberately cover broader abstract profiles.
All nine public correctness specifications (including the five preserved
historical declarations) and four public FP equivalence theorems pass explicit
axiom audits with only `propext`, `Classical.choice` and `Quot.sound`. The FP proofs
require no numerical-rule admission. The integrated `Correct`/`FPEquiv` build
also checks the migrated historical helpers.

`Proofs/RealExecution.lean` checks successful executions of all four complete
multi-tile source branches at `N=3`, `TN=2`. Uniform zero forward inputs yield
`[1/3, 1/3, 1/3]`. Backward active inputs and gradients equal to 1, with undefined
padding equal to 2, yield `[-6, -6, -6]`, confirming the padded product contributes
4 to the scale. Each witness checks that the next output cell remains 7. These
concrete executions and selected sentinel checks do not replace the symbolic IO,
access-safety and full-frame obligations.

The strict three-entry integrated build passed all 3536 jobs after migration.
All five historical public specifications remain alongside the four new real
and four FP contracts. All thirteen contracts are registered with current
source/support hashes and `test-gap` status; independent comparator replay
is still pending.
